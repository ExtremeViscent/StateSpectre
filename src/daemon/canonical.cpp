// v2 canonical model-state object store: RPC handlers + helpers.
//
// Design: a canonical object does not re-implement tiering. Its bytes are
// backed by the existing v1 slot/lease/drain/cold machinery through a synthetic
// tensor_id (canonical_tid(object_id)). The create path therefore mirrors
// handle_request_offload's slot reservation, and the rank then drives the
// unchanged MarkD2HSubmitted -> (D2H) -> MarkD2HComplete opcodes against the
// synthetic tid; those transition the slot to PINNED_VALID, publish
// locations_[synthetic_tid], and may enqueue a background drain exactly as in
// v1. CommitCanonicalObject finalizes the object metadata (state, content
// hash) and resolves duplicate-candidate / hash-verify races.
//
// See 01_CANONICAL_MODEL_STATE_STORE.md, 02_DEDUPLICATION_POLICY.md,
// 03_ROLLOUT_PULL_AND_EXPORT.md, 04_MULTI_JOB_NAMESPACE_AND_QUOTAS.md.

#include "daemon.h"

#include <cstring>

#include "metrics.h"
#include "util.h"
#include "wire_v2.h"

namespace offload {

namespace {
constexpr const char* TAG = "canonical";

// Candidate synthetic-tid space: bit 62 distinguishes throwaway duplicate/
// verify candidates from committed canonical objects so their location-table
// churn never clobbers the winner's canonical location entry.
constexpr uint64_t kCandidateBit = (1ull << 62);
inline uint64_t candidate_tid(uint64_t seq) {
    return kCanonicalTidBit | kCandidateBit | seq;
}

void append_u64(std::string* s, uint64_t v) {
    char buf[21];
    int n = std::snprintf(buf, sizeof(buf), "%llu:", (unsigned long long)v);
    s->append(buf, n);
}
void append_i64(std::string* s, int64_t v) {
    char buf[24];
    int n = std::snprintf(buf, sizeof(buf), "%lld:", (long long)v);
    s->append(buf, n);
}
}  // namespace

// ---------------------------------------------------------------------------
// Identity + hashing helpers
// ---------------------------------------------------------------------------

// Serialize the identity-critical axes of a canonical key. Deliberately
// EXCLUDES dp/replica axes (they are never part of the wire key — the Python
// key builder drops them) and job_name (display only). Two tensors that differ
// only in a replicated axis therefore produce the same string and dedup.
std::string OffloadDaemon::canonical_key_string(const CanonicalTensorKeyWire& k) {
    std::string s;
    s.reserve(160);
    append_u64(&s, k.job.tenant_id);
    append_u64(&s, k.job.job_id);
    append_u64(&s, k.job.launch_epoch);
    append_u64(&s, k.model_role);
    append_u64(&s, k.model_version);
    append_u64(&s, k.param_id);
    append_u64(&s, k.param_fqn_hash);
    append_u64(&s, k.dtype);
    append_u64(&s, k.layout);
    append_u64(&s, k.nbytes);
    append_u64(&s, k.shape_hash);
    append_u64(&s, k.stride_hash);
    append_u64(&s, k.pp_rank);
    append_u64(&s, k.tp_rank);
    append_u64(&s, k.ep_rank);
    append_u64(&s, k.etp_rank);
    append_i64(&s, k.expert_id);
    append_u64(&s, k.shard_offset);
    append_u64(&s, k.shard_nbytes);
    return s;
}

// 128-bit content hash: two FNV-1a streams with distinct seeds. Cheap, stable,
// and good enough to catch a corrupted replica (02_DEDUPLICATION_POLICY.md).
void OffloadDaemon::hash_bytes(const void* p, uint64_t n, uint64_t* lo, uint64_t* hi) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    uint64_t h1 = 1469598103934665603ull;   // FNV offset basis
    uint64_t h2 = 1099511628211ull;          // seeded differently
    for (uint64_t i = 0; i < n; ++i) {
        h1 = (h1 ^ b[i]) * 1099511628211ull;
        h2 = (h2 ^ b[i]) * 1099511628211ull;
        h2 = (h2 << 1) | (h2 >> 63);
    }
    *lo = h1;
    *hi = h2;
}

// ---------------------------------------------------------------------------
// Table lookups
// ---------------------------------------------------------------------------
JobRecord* OffloadDaemon::find_job(const JobUID& uid) {
    auto it = jobs_.find(uid);
    return it == jobs_.end() ? nullptr : &it->second;
}
JobRecord* OffloadDaemon::find_job_by_wire(const JobKeyWire& jk) {
    JobUID uid{jk.tenant_id, jk.job_id, jk.launch_epoch};
    return find_job(uid);
}
CanonicalObject* OffloadDaemon::find_object(uint64_t object_id) {
    auto it = objects_.find(object_id);
    return it == objects_.end() ? nullptr : &it->second;
}
CanonicalObject* OffloadDaemon::find_object_by_key(const std::string& key_str) {
    auto it = object_by_key_.find(key_str);
    if (it == object_by_key_.end()) return nullptr;
    return find_object(it->second);
}

bool OffloadDaemon::quota_allows_pinned(const JobRecord& job, uint64_t nbytes) const {
    if (job.max_pinned_bytes == 0) return true;  // unlimited
    return job.pinned_bytes + nbytes <= job.max_pinned_bytes;
}

// ---------------------------------------------------------------------------
// RegisterJob
// ---------------------------------------------------------------------------
RegisterJobResponse OffloadDaemon::handle_register_job(const RegisterJobRequest& req) {
    std::lock_guard<std::mutex> lk(mu_);
    RegisterJobResponse resp;
    if (!cfg_.v2_enable_canonical_store) {
        resp.ok = false;
        resp.message = "canonical store disabled";
        return resp;
    }

    JobRecord jr;
    jr.uid.tenant_id = req.tenant_id;
    // job_id: scheduler-provided stable id wins; else hash (tenant, name) so
    // the same logical job maps to the same id, but a RELAUNCH still gets a
    // fresh launch_epoch and thus a distinct namespace.
    if (req.scheduler_job_id != 0) {
        jr.uid.job_id = req.scheduler_job_id;
    } else {
        uint64_t lo = 0, hi = 0;
        std::string s = std::to_string(req.tenant_id) + "/" + req.job_name;
        hash_bytes(s.data(), s.size(), &lo, &hi);
        jr.uid.job_id = lo ? lo : 1;
    }
    jr.uid.launch_epoch = cfg_.v2_include_launch_epoch ? next_launch_epoch_++ : 1;
    jr.job_name = req.job_name;
    jr.state = JobState::kRunning;
    jr.created_ns = now_real_ns();
    jr.heartbeat_ns = jr.created_ns;
    // Seed per-job quotas from the daemon defaults.
    jr.max_pinned_bytes = cfg_.v2_quota_max_pinned_bytes;
    jr.max_pageable_bytes = cfg_.v2_quota_max_pageable_bytes;
    jr.max_nvme_bytes = cfg_.v2_quota_max_nvme_bytes;
    jr.max_inflight_d2h_bytes = cfg_.v2_quota_max_inflight_d2h_bytes;
    jr.max_inflight_export_bytes = cfg_.v2_quota_max_inflight_export_bytes;

    jobs_[jr.uid] = jr;
    Metrics::instance().inc(Metric::kJobsRegistered);

    resp.ok = true;
    resp.message = "ok";
    resp.job.tenant_id = jr.uid.tenant_id;
    resp.job.job_id = jr.uid.job_id;
    resp.job.launch_epoch = jr.uid.launch_epoch;
    resp.job.job_name = jr.job_name;
    resp.control_generation = header_ ? header_->generation : 0;
    OFLD_INFO(TAG, "RegisterJob tenant=%llu name=%s -> job_id=%llu epoch=%llu",
              (unsigned long long)jr.uid.tenant_id, jr.job_name.c_str(),
              (unsigned long long)jr.uid.job_id,
              (unsigned long long)jr.uid.launch_epoch);
    return resp;
}

// ---------------------------------------------------------------------------
// RequestCanonicalEvict: attach-or-create dedup.
// ---------------------------------------------------------------------------
RequestCanonicalEvictResponse OffloadDaemon::handle_request_canonical_evict(
    const RequestCanonicalEvictRequest& req) {
    std::unique_lock<std::mutex> lk(mu_);
    RequestCanonicalEvictResponse resp;
    resp.action = ATTACH_ACTION_ERROR;

    if (!cfg_.v2_enable_canonical_store) {
        resp.ok = false; resp.message = "canonical store disabled"; return resp;
    }
    if (!epoch_valid(req.rank_id, req.rank_epoch)) {
        Metrics::instance().inc(Metric::kEpochMismatchRejected);
        resp.ok = false; resp.message = "epoch invalid"; return resp;
    }
    JobRecord* job = find_job_by_wire(req.key.job);
    if (!job) {
        resp.ok = false; resp.message = "job not registered"; return resp;
    }
    // Reject writes to an already-sealed version (immutability invariant).
    {
        VersionKey vk{job->uid, req.key.model_role, req.key.model_version};
        auto mit = manifests_.find(vk);
        if (mit != manifests_.end() && mit->second.state == MANIFEST_SEALED) {
            Metrics::instance().inc(Metric::kCanonicalStaleVersionReject);
            resp.action = ATTACH_ACTION_REJECTED_STALE_VERSION;
            resp.ok = false; resp.message = "model_version already sealed";
            return resp;
        }
    }

    const std::string key_str = canonical_key_string(req.key);
    CanonicalObject* existing = find_object_by_key(key_str);
    const uint32_t mode = req.dedup_mode;

    // ---- shard/dtype consistency check against an existing object ----
    if (existing && existing->nbytes != req.key.nbytes) {
        resp.ok = false; resp.message = "nbytes mismatch vs existing object";
        return resp;
    }

    // Helper: reserve a pinned run for a (synthetic_tid, version) and fill the
    // lease/slot response. Mirrors handle_request_offload's reservation block.
    auto reserve = [&](uint64_t synthetic_tid, uint64_t version,
                       uint64_t* lease_out, uint32_t* base_out,
                       uint32_t* count_out) -> bool {
        if (!quota_allows_pinned(*job, req.key.nbytes)) {
            Metrics::instance().inc(Metric::kQuotaRejections);
            return false;
        }
        uint32_t count = slots_for_bytes(req.key.nbytes);
        uint32_t base = 0xFFFFFFFFu, ncount = 0;
        bool blocked = false, got = false;
        for (int attempt = 0; attempt <= 64; ++attempt) {
            if (allocate_slots(req.gpu_id, req.numa_node, req.key.nbytes,
                               OFLD_FLAG_DESTRUCTIVE, /*for_prefetch=*/false,
                               &base, &ncount, &blocked)) { got = true; break; }
            if (!force_drain_one(lk)) break;
        }
        if (!got) { Metrics::instance().inc(Metric::kBlockedReservations); return false; }
        uint64_t lease_id = next_lease_id_++;
        uint64_t owner_pid = sessions_[req.rank_id].pid;
        for (uint32_t k = 0; k < count; ++k) {
            uint32_t id = base + k;
            ofld_slot_entry_t& e = slots_[id];
            e.tensor_id = synthetic_tid;
            e.version = version;
            e.lease_id = lease_id;
            e.owner_rank = req.rank_id;
            e.owner_pid = owner_pid;
            e.rank_epoch = req.rank_epoch;
            e.nbytes = (k == 0) ? req.key.nbytes : e.capacity;
            e.flags = OFLD_FLAG_DESTRUCTIVE;
            set_slot_state(id, OFLD_SLOT_RESERVED_D2H);
        }
        Metrics::instance().add(Metric::kUsedPinnedBytes,
                                static_cast<uint64_t>(count) *
                                    cfg_.allocation_granularity_bytes);
        Lease L;
        L.lease_id = lease_id; L.base_slot = base; L.slot_count = count;
        L.tensor_id = synthetic_tid; L.version = version;
        L.owner_rank = req.rank_id; L.owner_pid = owner_pid;
        L.rank_epoch = req.rank_epoch; L.nbytes = req.key.nbytes;
        L.is_prefetch = false;
        leases_[lease_id] = L;
        job->pinned_bytes += req.key.nbytes;
        *lease_out = lease_id; *base_out = base; *count_out = count;
        return true;
    };

    auto fill_slot_resp = [&](uint32_t base) {
        resp.slot_id = base;
        resp.arena_id = slots_[base].arena_id;
        resp.arena_offset = slots_[base].arena_offset;
        resp.capacity = static_cast<uint64_t>(slots_for_bytes(req.key.nbytes)) *
                        cfg_.allocation_granularity_bytes;
    };

    // ======================= existing VALID/SEALED object ==================
    if (existing && (existing->state == OBJ_VALID_PINNED ||
                     existing->state == OBJ_VALID_PAGEABLE ||
                     existing->state == OBJ_VALID_NVME ||
                     existing->state == OBJ_VALID_MULTI_TIER ||
                     existing->state == OBJ_SEALED ||
                     existing->state == OBJ_EXPORT_IN_FLIGHT)) {
        if (!req.attach_if_exists) {
            resp.ok = false; resp.message = "exists but attach disabled"; return resp;
        }
        const bool need_verify = (mode == DEDUP_HASH_VERIFIED ||
                                  mode == DEDUP_SAMPLED_HASH);
        if (!need_verify) {
            // Semantic trusted / disabled-but-present: attach, no D2H.
            existing->refcount++;
            existing->last_access_ns = now_real_ns();
            resp.ok = true;
            resp.action = ATTACH_ACTION_ATTACHED_EXISTING;
            resp.object_id = existing->object_id;
            resp.message = "attached existing";
            Metrics::instance().inc(Metric::kCanonicalAttachExisting);
            Metrics::instance().add(Metric::kCanonicalDedupBytesSaved, req.key.nbytes);
            Metrics::instance().add(Metric::kCanonicalD2HBytesAvoided, req.key.nbytes);
            return resp;
        }
        // Hash-verified attach: the rank must D2H into a throwaway candidate so
        // the daemon can compare bytes on commit. Give it a candidate slot.
        uint64_t seq = next_object_id_++;  // reuse counter for a unique cand tid
        uint64_t cand_tid = candidate_tid(seq);
        uint64_t lease_id; uint32_t base, count;
        if (!reserve(cand_tid, /*version=*/1, &lease_id, &base, &count)) {
            resp.action = ATTACH_ACTION_QUOTA_EXCEEDED;
            resp.ok = false; resp.message = "no capacity for verify candidate";
            return resp;
        }
        resp.ok = true;
        resp.action = ATTACH_ACTION_DUPLICATE_CANDIDATE;
        resp.object_id = existing->object_id;   // verify against this object
        resp.lease_id = lease_id;
        fill_slot_resp(base);
        resp.message = "verify candidate (hash)";
        Metrics::instance().inc(Metric::kCanonicalDuplicateCandidate);
        return resp;
    }

    // ======================= object is CREATING ============================
    if (existing && existing->state == OBJ_CREATING) {
        const bool pressure_hi =
            pinned_pressure() >= cfg_.v2_duplicate_candidate_gpu_pressure_threshold;
        const bool want_dup =
            req.allow_duplicate_candidate ||
            (cfg_.v2_creating_policy == 1 && pressure_hi);
        if (!want_dup) {
            // WAIT_FOR_CREATOR: the client re-issues RequestCanonicalEvict and
            // gets ATTACHED_EXISTING once the creator commits. Lock-friendly.
            resp.ok = true;
            resp.action = ATTACH_ACTION_WAIT_FOR_CREATOR;
            resp.object_id = existing->object_id;
            resp.message = "wait for creator";
            Metrics::instance().inc(Metric::kCanonicalAttachWait);
            return resp;
        }
        // ALLOW_DUPLICATE_CANDIDATE: hand out a separate candidate slot; the
        // first commit wins, the loser discards on commit.
        uint64_t seq = next_object_id_++;
        uint64_t cand_tid = candidate_tid(seq);
        uint64_t lease_id; uint32_t base, count;
        if (!reserve(cand_tid, 1, &lease_id, &base, &count)) {
            resp.action = ATTACH_ACTION_QUOTA_EXCEEDED;
            resp.ok = false; resp.message = "no capacity for duplicate candidate";
            return resp;
        }
        resp.ok = true;
        resp.action = ATTACH_ACTION_DUPLICATE_CANDIDATE;
        resp.object_id = existing->object_id;
        resp.lease_id = lease_id;
        fill_slot_resp(base);
        resp.message = "duplicate candidate";
        Metrics::instance().inc(Metric::kCanonicalDuplicateCandidate);
        return resp;
    }

    // ======================= missing: create ===============================
    if (!req.create_if_missing) {
        resp.ok = false; resp.message = "missing and create disabled"; return resp;
    }
    uint64_t object_id = next_object_id_++;
    uint64_t synthetic_tid = canonical_tid(object_id);
    uint64_t lease_id; uint32_t base, count;
    if (!reserve(synthetic_tid, /*version=*/1, &lease_id, &base, &count)) {
        resp.action = ATTACH_ACTION_QUOTA_EXCEEDED;
        resp.ok = false; resp.message = "no pinned capacity / quota";
        return resp;
    }

    CanonicalObject obj;
    obj.object_id = object_id;
    obj.key = req.key;
    obj.key_str = key_str;
    obj.nbytes = req.key.nbytes;
    obj.state = OBJ_CREATING;
    obj.refcount = 1;
    obj.producer_rank = req.rank_id;
    obj.synthetic_tid = synthetic_tid;
    obj.version = 1;
    obj.lease_id = lease_id;
    obj.base_slot = base;
    obj.slot_count = count;
    obj.creating_rank = req.rank_id;
    obj.created_ns = now_real_ns();
    obj.last_access_ns = obj.created_ns;
    objects_[object_id] = obj;
    object_by_key_[key_str] = object_id;
    job->canonical_object_count++;

    resp.ok = true;
    resp.action = ATTACH_ACTION_NEED_D2H_CREATE;
    resp.object_id = object_id;
    resp.lease_id = lease_id;
    fill_slot_resp(base);
    resp.message = "need d2h create";
    Metrics::instance().inc(Metric::kCanonicalObjectsCreated);
    return resp;
}

// ---------------------------------------------------------------------------
// CommitCanonicalObject: finalize creator, or resolve candidate/verify.
// ---------------------------------------------------------------------------
CommitCanonicalObjectResponse OffloadDaemon::handle_commit_canonical_object(
    const CommitCanonicalObjectRequest& req) {
    std::unique_lock<std::mutex> lk(mu_);
    CommitCanonicalObjectResponse resp;
    resp.object_id = req.object_id;
    resp.won = false;

    if (!cfg_.v2_enable_canonical_store) {
        resp.ok = false; resp.message = "canonical store disabled"; return resp;
    }
    if (!epoch_valid(req.rank_id, req.rank_epoch)) {
        Metrics::instance().inc(Metric::kEpochMismatchRejected);
        resp.ok = false; resp.message = "epoch invalid"; return resp;
    }
    CanonicalObject* obj = find_object(req.object_id);
    if (!obj) { resp.ok = false; resp.message = "unknown object"; return resp; }

    Lease* L = find_lease(req.lease_id);
    if (!L) { resp.ok = false; resp.message = "unknown lease"; return resp; }

    const bool is_creator =
        (obj->state == OBJ_CREATING && obj->lease_id == req.lease_id);

    // ---------------- creator commit ----------------
    if (is_creator) {
        // The rank's MarkD2HComplete has already driven the slot to
        // PINNED_VALID and published locations_[synthetic_tid]. Finalize state.
        uint32_t st = slot_state(obj->base_slot);
        if (st != OFLD_SLOT_PINNED_VALID && st != OFLD_SLOT_DRAIN_IN_FLIGHT &&
            st != OFLD_SLOT_COLD_VALID) {
            Metrics::instance().inc(Metric::kInvalidTransitionRejected);
            resp.ok = false; resp.message = "object bytes not yet valid"; return resp;
        }
        obj->content_hash_lo = req.content_hash_lo;
        obj->content_hash_hi = req.content_hash_hi;
        obj->state = OBJ_VALID_PINNED;
        obj->creating_rank = 0xFFFFFFFFu;
        obj->last_access_ns = now_real_ns();
        resp.ok = true; resp.won = true; resp.message = "committed";
        return resp;
    }

    // ---------------- candidate / verify commit ----------------
    // This lease belongs to a throwaway candidate slot (its own synthetic tid).
    // Compare against the (now-existing) canonical object, then discard.
    const bool obj_valid = (obj->state != OBJ_CREATING && obj->state != OBJ_MISSING &&
                            obj->state != OBJ_ERROR && obj->state != OBJ_DELETED);

    // Hash verification: if the object carries a content hash and the candidate
    // provided one, they must match.
    bool hash_ok = true;
    if (obj_valid && (obj->content_hash_lo != 0 || obj->content_hash_hi != 0) &&
        (req.content_hash_lo != 0 || req.content_hash_hi != 0)) {
        hash_ok = (obj->content_hash_lo == req.content_hash_lo &&
                   obj->content_hash_hi == req.content_hash_hi);
    }

    // Free the candidate's slot run + lease + throwaway location.
    uint32_t cbase = L->base_slot, ccnt = L->slot_count;
    uint64_t ctid = L->tensor_id;
    JobRecord* job = find_job_by_wire(obj->key.job);
    for (uint32_t k = 0; k < ccnt; ++k) {
        uint32_t st = slot_state(cbase + k);
        if (st == OFLD_SLOT_DRAIN_IN_FLIGHT || st == OFLD_SLOT_COLD_VALID) {
            // Drain owns the slot lifecycle; just drop the throwaway location.
        }
    }
    // Only free if not mid-drain (mirror release_lease semantics).
    uint32_t cst = slot_state(cbase);
    if (cst == OFLD_SLOT_PINNED_VALID || cst == OFLD_SLOT_RESERVED_D2H ||
        cst == OFLD_SLOT_D2H_IN_FLIGHT) {
        free_slots(cbase, ccnt);
    }
    locations_.erase(ctid);
    leases_.erase(req.lease_id);
    if (job && job->pinned_bytes >= obj->nbytes) job->pinned_bytes -= obj->nbytes;

    if (!obj_valid) {
        resp.ok = false; resp.message = "canonical object not valid for attach";
        return resp;
    }
    if (!hash_ok) {
        Metrics::instance().inc(Metric::kCanonicalHashMismatch);
        Metrics::instance().inc(Metric::kChecksumMismatch);
        resp.ok = false; resp.won = false;
        resp.message = "content hash mismatch (corrupt replica)";
        return resp;
    }
    // Verified/duplicate loser: attach to the winner.
    obj->refcount++;
    obj->last_access_ns = now_real_ns();
    resp.ok = true; resp.won = false; resp.message = "attached after verify";
    Metrics::instance().inc(Metric::kCanonicalAttachExisting);
    Metrics::instance().add(Metric::kCanonicalDedupBytesSaved, obj->nbytes);
    return resp;
}

// ---------------------------------------------------------------------------
// Release a canonical object's physical backing + table entry (mu_ held).
// ---------------------------------------------------------------------------
void OffloadDaemon::release_object(CanonicalObject& obj) {
    if (obj.export_refcount > 0) return;  // export lease holds it
    auto lit = locations_.find(obj.synthetic_tid);
    if (lit != locations_.end()) {
        if (lit->second.kind == OFLD_LOC_PINNED &&
            obj.base_slot != 0xFFFFFFFFu) {
            uint32_t st = slot_state(obj.base_slot);
            if (st == OFLD_SLOT_PINNED_VALID)
                free_slots(obj.base_slot, obj.slot_count);
        } else if (lit->second.cold_ref != 0) {
            cold_store_.erase(lit->second.cold_ref);
        }
        locations_.erase(lit);
    }
    leases_.erase(obj.lease_id);
    object_by_key_.erase(obj.key_str);
    JobRecord* job = find_job_by_wire(obj.key.job);
    if (job) {
        if (job->pinned_bytes >= obj.nbytes) job->pinned_bytes -= obj.nbytes;
        if (job->canonical_object_count > 0) job->canonical_object_count--;
    }
    objects_.erase(obj.object_id);
}

}  // namespace offload
