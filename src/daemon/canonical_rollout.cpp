// v2 canonical model-state: sealed manifests + rollout pull/export.
//
// Manifests make a model version immutable and rollout-visible. Rollout engines
// pull ONLY sealed versions (docs/design/rollout-pull-and-export.md). Export never races
// slot recycling/drain: it takes an export lease (export_refcount++), stages the
// object's current bytes into a host-owned buffer, transports it, then releases.
//
// Transport backends live in export_backend.{h,cpp}; this file locates the
// object, stages it, and dispatches to the backend chosen by the request.

#include "daemon.h"

#include <algorithm>
#include <cstring>
#include <functional>

#include "export_backend.h"
#include "metrics.h"
#include "util.h"
#include "wire_v2.h"

namespace offload {

namespace {
constexpr const char* TAG = "canonical";

const char* loc_hint(ofld_location_kind_t k) {
    switch (k) {
        case OFLD_LOC_PINNED:   return "pinned";
        case OFLD_LOC_PAGEABLE: return "pageable";
        case OFLD_LOC_NVME:     return "nvme";
        case OFLD_LOC_GPU:      return "gpu";
        default:                return "none";
    }
}
}  // namespace

// ---------------------------------------------------------------------------
// SealModelVersion (+ optional atomic promote of the latest-sealed pointer).
// ---------------------------------------------------------------------------
SealModelVersionResponse OffloadDaemon::handle_seal_model_version(
    const SealModelVersionRequest& req) {
    std::lock_guard<std::mutex> lk(mu_);
    SealModelVersionResponse resp;
    resp.state = MANIFEST_ERROR;

    if (!cfg_.v2_enable_canonical_store || !cfg_.v2_enable_manifests) {
        resp.ok = false; resp.message = "manifests disabled"; return resp;
    }
    JobRecord* job = find_job_by_wire(req.job);
    if (!job) { resp.ok = false; resp.message = "job not registered"; return resp; }

    JobUID uid = job->uid;
    VersionKey vk{uid, req.model_role, req.model_version};

    // Idempotent: already sealed.
    auto mit = manifests_.find(vk);
    if (mit != manifests_.end() && mit->second.state == MANIFEST_SEALED) {
        resp.ok = true; resp.state = MANIFEST_SEALED;
        resp.tensor_count = mit->second.object_ids.size();
        resp.total_nbytes = mit->second.total_nbytes;
        resp.manifest_hash_lo = mit->second.manifest_hash_lo;
        resp.manifest_hash_hi = mit->second.manifest_hash_hi;
        resp.message = "already sealed";
        return resp;
    }

    // Gather all VALID objects matching (job, role, version).
    Manifest man;
    man.job = uid;
    man.model_role = req.model_role;
    man.model_version = req.model_version;
    man.state = MANIFEST_SEALING;
    man.created_ns = now_real_ns();

    // Manifest hash accumulates each member's (object_id, content_hash, nbytes).
    uint64_t mh_lo = 1469598103934665603ull, mh_hi = 1099511628211ull;
    auto mix = [&](uint64_t v) {
        mh_lo = (mh_lo ^ v) * 1099511628211ull;
        mh_hi = (mh_hi ^ (v + 0x9e3779b97f4a7c15ull)) * 1099511628211ull;
    };

    bool any_invalid = false;
    for (auto& kv : objects_) {
        CanonicalObject& o = kv.second;
        if (!(o.key.job.tenant_id == uid.tenant_id &&
              o.key.job.job_id == uid.job_id &&
              o.key.job.launch_epoch == uid.launch_epoch)) continue;
        if (o.key.model_role != req.model_role) continue;
        if (o.key.model_version != req.model_version) continue;

        bool valid = (o.state == OBJ_VALID_PINNED || o.state == OBJ_VALID_PAGEABLE ||
                      o.state == OBJ_VALID_NVME || o.state == OBJ_VALID_MULTI_TIER ||
                      o.state == OBJ_SEALED || o.state == OBJ_EXPORT_IN_FLIGHT);
        if (!valid) { any_invalid = true; continue; }
        man.object_ids.push_back(o.object_id);
        man.total_nbytes += o.nbytes;
    }

    // Enforce expected_param_ids coverage if the caller supplied a set.
    if (!req.expected_param_ids.empty()) {
        for (uint64_t pid : req.expected_param_ids) {
            bool found = false;
            for (uint64_t oid : man.object_ids) {
                CanonicalObject* o = find_object(oid);
                if (o && o->key.param_id == pid) { found = true; break; }
            }
            if (!found) { any_invalid = true; break; }
        }
    }

    if (cfg_.v2_seal_requires_all_objects_valid && req.fail_if_missing && any_invalid) {
        man.state = MANIFEST_ERROR;
        manifests_[vk] = man;
        Metrics::instance().inc(Metric::kManifestSealFailed);
        resp.ok = false; resp.state = MANIFEST_ERROR;
        resp.message = "seal failed: required objects missing/invalid";
        return resp;
    }

    // Sort members by object_id for a deterministic manifest hash.
    std::sort(man.object_ids.begin(), man.object_ids.end());
    for (uint64_t oid : man.object_ids) {
        CanonicalObject* o = find_object(oid);
        if (!o) continue;
        mix(oid); mix(o->content_hash_lo); mix(o->content_hash_hi); mix(o->nbytes);
        o->state = OBJ_SEALED;   // freeze member bytes
        o->sealed_ns = now_real_ns();
    }
    man.manifest_hash_lo = mh_lo;
    man.manifest_hash_hi = mh_hi;
    man.state = MANIFEST_SEALED;
    man.sealed_ns = now_real_ns();
    manifests_[vk] = man;
    Metrics::instance().inc(Metric::kManifestSealed);

    // Atomic promote of the rollout latest pointer.
    if (req.promote) {
        RoleKey rk{uid, req.model_role};
        latest_sealed_[rk] = req.model_version;
        OFLD_INFO(TAG, "promoted job=%llu role=%u -> latest sealed version=%llu",
                  (unsigned long long)uid.job_id, req.model_role,
                  (unsigned long long)req.model_version);
        // Bound the rollout-publish path: drop sealed versions beyond the
        // retention window (oldest first, never the just-promoted one).
        enforce_sealed_retention(uid, req.model_role);
    }

    resp.ok = true;
    resp.state = MANIFEST_SEALED;
    resp.tensor_count = man.object_ids.size();
    resp.total_nbytes = man.total_nbytes;
    resp.manifest_hash_lo = mh_lo;
    resp.manifest_hash_hi = mh_hi;
    resp.message = "sealed";
    OFLD_INFO(TAG, "SealModelVersion job=%llu role=%u version=%llu tensors=%llu",
              (unsigned long long)uid.job_id, req.model_role,
              (unsigned long long)req.model_version,
              (unsigned long long)man.object_ids.size());
    return resp;
}

// ---------------------------------------------------------------------------
// GetLatestSealedVersion
// ---------------------------------------------------------------------------
GetLatestSealedVersionResponse OffloadDaemon::handle_get_latest_sealed_version(
    const GetLatestSealedVersionRequest& req) {
    std::lock_guard<std::mutex> lk(mu_);
    GetLatestSealedVersionResponse resp;
    JobRecord* job = find_job_by_wire(req.job);
    if (!job) { resp.ok = false; resp.message = "job not registered"; return resp; }
    RoleKey rk{job->uid, req.model_role};
    auto it = latest_sealed_.find(rk);
    resp.ok = true;
    if (it == latest_sealed_.end()) {
        resp.found = false; resp.message = "no sealed version promoted";
    } else {
        resp.found = true; resp.model_version = it->second; resp.message = "ok";
    }
    return resp;
}

// ---------------------------------------------------------------------------
// GetManifest
// ---------------------------------------------------------------------------
GetManifestResponse OffloadDaemon::handle_get_manifest(const GetManifestRequest& req) {
    std::lock_guard<std::mutex> lk(mu_);
    GetManifestResponse resp;
    resp.state = MANIFEST_ERROR;
    JobRecord* job = find_job_by_wire(req.job);
    if (!job) { resp.ok = false; resp.message = "job not registered"; return resp; }

    resp.job = req.job;
    resp.model_role = req.model_role;
    resp.model_version = req.model_version;

    VersionKey vk{job->uid, req.model_role, req.model_version};
    auto it = manifests_.find(vk);
    if (it == manifests_.end() || it->second.state != MANIFEST_SEALED) {
        Metrics::instance().inc(Metric::kRolloutStaleManifestReject);
        resp.ok = false; resp.message = "no sealed manifest for version";
        return resp;
    }
    const Manifest& man = it->second;
    resp.state = man.state;

    const ManifestFilterWire& f = req.filter;
    for (uint64_t oid : man.object_ids) {
        CanonicalObject* o = find_object(oid);
        if (!o) continue;
        const CanonicalTensorKeyWire& k = o->key;
        if (f.pp_present && k.pp_rank != f.pp_rank) continue;
        if (f.tp_present && k.tp_rank != f.tp_rank) continue;
        if (f.ep_present && k.ep_rank != f.ep_rank) continue;
        if (f.etp_present && k.etp_rank != f.etp_rank) continue;
        if (f.expert_present && k.expert_id != f.expert_id) continue;

        TensorManifestEntryWire e;
        e.key = k;
        e.object_id = oid;
        e.nbytes = o->nbytes;
        e.content_hash_lo = o->content_hash_lo;
        e.content_hash_hi = o->content_hash_hi;
        auto lit = locations_.find(o->synthetic_tid);
        e.location_hint = loc_hint(lit != locations_.end() ? lit->second.kind
                                                           : OFLD_LOC_NONE);
        resp.tensors.push_back(std::move(e));
    }
    resp.ok = true;
    resp.message = "ok";
    Metrics::instance().inc(Metric::kRolloutGetManifest);
    return resp;
}

// ---------------------------------------------------------------------------
// stage_object_for_export: resolve current physical bytes into a host buffer.
// May unlock lk for NVMe/pageable IO. Caller must hold an export lease so the
// backing cannot be recycled while unlocked.
// ---------------------------------------------------------------------------
std::vector<uint8_t> OffloadDaemon::stage_object_for_export(
    std::unique_lock<std::mutex>& lk, CanonicalObject& obj, bool* ok) {
    *ok = false;
    std::vector<uint8_t> buf;
    uint64_t nbytes = obj.nbytes;
    auto lit = locations_.find(obj.synthetic_tid);
    if (lit == locations_.end()) return buf;
    Location loc = lit->second;

    if (loc.kind == OFLD_LOC_PINNED) {
        // Copy straight out of the pinned slot (fast path).
        void* src = slot_addr(loc.slot_id);
        buf.resize(nbytes);
        std::memcpy(buf.data(), src, nbytes);
        Metrics::instance().inc(Metric::kExportFromPinned);
        *ok = true;
        return buf;
    }
    if (loc.kind == OFLD_LOC_PAGEABLE || loc.kind == OFLD_LOC_NVME) {
        ColdEntry cold;
        auto cit = cold_store_.find(loc.cold_ref);
        if (cit == cold_store_.end()) return buf;
        cold = cit->second;
        buf.resize(nbytes);
        // Cold read may block; drop the lock (export lease protects lifetime).
        lk.unlock();
        bool rok = do_readback_from_cold(/*base_slot=*/0, /*slot_count=*/0,
                                         cold, nbytes);
        // do_readback_from_cold writes into a slot; for export we instead read
        // the cold bytes directly into buf via the cold entry.
        (void)rok;
        bool copied = false;
        if (cold.kind == OFLD_LOC_PAGEABLE && cold.pageable) {
            std::memcpy(buf.data(), cold.pageable, nbytes);
            copied = true;
        } else if (cold.kind == OFLD_LOC_NVME) {
            copied = nvme_read(cold, buf.data(), nbytes);
        }
        lk.lock();
        if (!copied) return buf;
        Metrics::instance().inc(loc.kind == OFLD_LOC_NVME
                                    ? Metric::kExportFromNvme
                                    : Metric::kExportFromPageable);
        *ok = true;
        return buf;
    }
    return buf;
}

// ---------------------------------------------------------------------------
// PullTensor: rollout-triggered export of a sealed canonical object.
// ---------------------------------------------------------------------------
PullTensorResponse OffloadDaemon::handle_pull_tensor(const PullTensorRequest& req) {
    std::unique_lock<std::mutex> lk(mu_);
    PullTensorResponse resp;
    resp.object_id = req.object_id;
    resp.transport = req.transport;

    if (!cfg_.v2_enable_canonical_store || !cfg_.v2_enable_rollout_export) {
        resp.ok = false; resp.message = "rollout export disabled"; return resp;
    }
    JobRecord* job = find_job_by_wire(req.job);
    if (!job) { resp.ok = false; resp.message = "job not registered"; return resp; }

    // Rollout may pull ONLY sealed versions (critical invariant).
    VersionKey vk{job->uid, req.model_role, req.model_version};
    auto mit = manifests_.find(vk);
    if (mit == manifests_.end() || mit->second.state != MANIFEST_SEALED) {
        Metrics::instance().inc(Metric::kRolloutStaleManifestReject);
        resp.ok = false; resp.message = "version not sealed; refusing pull";
        return resp;
    }
    CanonicalObject* obj = find_object(req.object_id);
    if (!obj) { resp.ok = false; resp.message = "unknown object"; return resp; }
    if (obj->state != OBJ_SEALED) {
        resp.ok = false; resp.message = "object not sealed"; return resp;
    }
    // Object must be a member of the sealed manifest.
    {
        bool member = false;
        for (uint64_t oid : mit->second.object_ids)
            if (oid == req.object_id) { member = true; break; }
        if (!member) { resp.ok = false; resp.message = "object not in manifest"; return resp; }
    }

    // Per-job concurrent export cap.
    if (cfg_.v2_max_concurrent_exports_per_job > 0 &&
        job->inflight_export_bytes > 0) {
        // Soft cap by count is tracked via export_refcount across objects; here
        // we bound inflight bytes-per-job only if a hard quota is set.
        if (job->max_inflight_export_bytes > 0 &&
            job->inflight_export_bytes + obj->nbytes > job->max_inflight_export_bytes) {
            Metrics::instance().inc(Metric::kQuotaRejections);
            resp.ok = false; resp.message = "export inflight quota exceeded";
            return resp;
        }
    }

    // Take the export lease BEFORE staging so the backing can't be recycled.
    obj->export_refcount++;
    uint32_t prev_state = obj->state;
    obj->state = OBJ_EXPORT_IN_FLIGHT;
    job->inflight_export_bytes += obj->nbytes;
    Metrics::instance().add(Metric::kExportInflightBytes, obj->nbytes);

    uint64_t nbytes = obj->nbytes;
    uint64_t hlo = obj->content_hash_lo, hhi = obj->content_hash_hi;

    bool staged_ok = false;
    std::vector<uint8_t> buf = stage_object_for_export(lk, *obj, &staged_ok);
    // obj may have been rehashed by pointer invalidation across unlock; re-find.
    obj = find_object(req.object_id);

    if (!staged_ok || !obj) {
        if (obj) {
            obj->export_refcount--;
            obj->state = static_cast<CanonicalObjectState>(prev_state);
        }
        if (job) job->inflight_export_bytes -= nbytes;
        Metrics::instance().sub(Metric::kExportInflightBytes, nbytes);
        Metrics::instance().inc(Metric::kExportTransportErrors);
        resp.ok = false; resp.message = "staging failed"; return resp;
    }
    Metrics::instance().add(Metric::kExportStagingBytes, nbytes);

    // Dispatch to the transport backend WITHOUT holding mu_ (network IO).
    ExportRequest xr;
    xr.object_id = req.object_id;
    xr.nbytes = nbytes;
    xr.transport = req.transport;
    xr.target_descriptor = req.target_descriptor;
    xr.content_hash_lo = hlo;
    xr.content_hash_hi = hhi;

    lk.unlock();
    ExportResult xres = export_send(xr, buf.data(), buf.size(),
                                    cfg_.v2_fallback_transport);
    lk.lock();

    // Release the export lease.
    obj = find_object(req.object_id);
    if (obj) {
        if (obj->export_refcount > 0) obj->export_refcount--;
        if (obj->export_refcount == 0)
            obj->state = static_cast<CanonicalObjectState>(prev_state);
    }
    JobRecord* job2 = find_job_by_wire(req.job);
    if (job2 && job2->inflight_export_bytes >= nbytes)
        job2->inflight_export_bytes -= nbytes;
    Metrics::instance().sub(Metric::kExportInflightBytes, nbytes);
    Metrics::instance().sub(Metric::kExportStagingBytes, nbytes);
    // If every holder released while this export was in flight, reclaim now.
    if (obj) maybe_free_object(*obj);

    if (!xres.ok) {
        Metrics::instance().inc(Metric::kExportTransportErrors);
        resp.ok = false; resp.message = "transport failed: " + xres.message;
        return resp;
    }
    Metrics::instance().inc(Metric::kRolloutPullTensor);
    Metrics::instance().add(Metric::kRolloutPullBytes, nbytes);

    resp.ok = true;
    resp.object_id = req.object_id;
    resp.nbytes = nbytes;
    resp.transport = xres.transport;
    resp.content_hash_lo = hlo;
    resp.content_hash_hi = hhi;
    resp.transport_metadata = xres.transport_metadata;
    resp.message = "ok";
    return resp;
}

// ---------------------------------------------------------------------------
// RequestCanonicalRestore: make a canonical object readable in a pinned arena
// and hand back its (arena_id, arena_offset) so the requesting rank can H2D the
// bytes into its own GPU storage. This is the trainer offload/reload round-trip
// read path — any rank (including a replica that only ATTACHED, no local D2H)
// can read the one shared copy back. Strictly read-only: bumps restore_refcount
// so drain/recycle cannot free the backing during the H2D; never frees or
// migrates the object here.
// ---------------------------------------------------------------------------
RequestCanonicalRestoreResponse OffloadDaemon::handle_request_canonical_restore(
    const RequestCanonicalRestoreRequest& req) {
    std::unique_lock<std::mutex> lk(mu_);
    RequestCanonicalRestoreResponse resp;
    resp.object_id = req.object_id;

    if (!cfg_.v2_enable_canonical_store) {
        resp.ok = false; resp.message = "canonical store disabled"; return resp;
    }
    if (!epoch_valid(req.rank_id, req.rank_epoch)) {
        Metrics::instance().inc(Metric::kEpochMismatchRejected);
        resp.ok = false; resp.message = "epoch invalid"; return resp;
    }
    CanonicalObject* obj = find_object(req.object_id);
    if (!obj) { resp.ok = false; resp.message = "unknown object"; return resp; }
    const bool valid = (obj->state == OBJ_VALID_PINNED || obj->state == OBJ_VALID_PAGEABLE ||
                        obj->state == OBJ_VALID_NVME || obj->state == OBJ_VALID_MULTI_TIER ||
                        obj->state == OBJ_SEALED || obj->state == OBJ_EXPORT_IN_FLIGHT);
    if (!valid) { resp.ok = false; resp.message = "object not readable"; return resp; }

    const uint64_t syn_tid = canonical_tid(req.object_id);
    const uint64_t nbytes = obj->nbytes;
    auto lit = locations_.find(syn_tid);
    if (lit == locations_.end() || lit->second.kind == OFLD_LOC_NONE ||
        lit->second.kind == OFLD_LOC_DELETED) {
        resp.ok = false; resp.message = "object not resident"; return resp;
    }

    uint32_t base_slot = 0xFFFFFFFFu;
    if (lit->second.kind == OFLD_LOC_PINNED &&
        slot_state(lit->second.slot_id) == OFLD_SLOT_PINNED_VALID) {
        // Already pinned — read straight from the resident slot.
        base_slot = lit->second.slot_id;
    } else if (lit->second.kind == OFLD_LOC_PAGEABLE ||
               lit->second.kind == OFLD_LOC_NVME) {
        // Cold (pageable/NVMe): warm it back into a pinned slot. The warmed copy
        // becomes the object's pinned residency (multi-tier: cold_ref is kept),
        // so subsequent restores are warm.
        if (lit->second.cold_ref == 0 ||
            cold_store_.find(lit->second.cold_ref) == cold_store_.end()) {
            // Cold reference not yet settled (drain still finalizing) — retry.
            resp.ok = false; resp.retriable = true;
            resp.message = "object cold reference not ready";
            return resp;
        }
        ColdEntry cold = cold_store_[lit->second.cold_ref];
        uint32_t base = 0xFFFFFFFFu, count = 0;
        bool blocked = false, got = false;
        for (int attempt = 0; attempt <= 64; ++attempt) {
            if (allocate_slots(req.gpu_id, req.numa_node, nbytes,
                               OFLD_FLAG_ALLOW_OVERFLOW, /*for_prefetch=*/true,
                               &base, &count, &blocked)) { got = true; break; }
            if (!force_drain_one(lk)) break;
        }
        if (!got) {
            resp.ok = false; resp.message = "no pinned capacity to warm object";
            return resp;
        }
        // Bind the run to the object (synthetic tid) and read the cold bytes in.
        uint64_t owner_pid = sessions_[req.rank_id].pid;
        for (uint32_t k = 0; k < count; ++k) {
            ofld_slot_entry_t& e = slots_[base + k];
            e.tensor_id = syn_tid; e.version = obj->version;
            e.owner_rank = req.rank_id; e.owner_pid = owner_pid;
            e.rank_epoch = req.rank_epoch;
            e.nbytes = (k == 0) ? nbytes : e.capacity;
            set_slot_state(base + k, OFLD_SLOT_READBACK_IN_FLIGHT);
        }
        lk.unlock();
        bool rok = do_readback_from_cold(base, count, cold, nbytes);
        lk.lock();
        if (!rok) {
            for (uint32_t k = 0; k < count; ++k) set_slot_state(base + k, OFLD_SLOT_ERROR);
            free_slots(base, count);
            Metrics::instance().inc(Metric::kIoFailure);
            resp.ok = false; resp.message = "readback failed"; return resp;
        }
        for (uint32_t k = 0; k < count; ++k) set_slot_state(base + k, OFLD_SLOT_PINNED_VALID);
        Metrics::instance().add(Metric::kUsedPinnedBytes,
                                static_cast<uint64_t>(count) *
                                    cfg_.allocation_granularity_bytes);
        // Warmed copy is now the object's pinned residency (keep cold_ref → multi-tier).
        auto& loc = locations_[syn_tid];
        loc.kind = OFLD_LOC_PINNED; loc.slot_id = base; loc.nbytes = nbytes;
        obj->base_slot = base; obj->slot_count = count;
        obj->state = OBJ_VALID_MULTI_TIER;
        base_slot = base;
    } else {
        // kind==PINNED but slot not PINNED_VALID → the object is mid-transition
        // (drain/readback in flight). Ask the caller to retry shortly.
        resp.ok = false; resp.retriable = true;
        resp.message = "object transitioning between tiers";
        return resp;
    }

    // Hold the object resident for the H2D window (read-only guard).
    obj->restore_refcount++;
    obj->last_access_ns = now_real_ns();
    Metrics::instance().inc(Metric::kCanonicalRestoreServed);

    resp.ok = true;
    resp.message = "ok";
    resp.nbytes = nbytes;
    resp.arena_id = slots_[base_slot].arena_id;
    resp.arena_offset = slots_[base_slot].arena_offset;
    return resp;
}

ReleaseCanonicalRestoreResponse OffloadDaemon::handle_release_canonical_restore(
    const ReleaseCanonicalRestoreRequest& req) {
    std::lock_guard<std::mutex> lk(mu_);
    ReleaseCanonicalRestoreResponse resp;
    if (!epoch_valid(req.rank_id, req.rank_epoch)) {
        Metrics::instance().inc(Metric::kEpochMismatchRejected);
        resp.ok = false; resp.message = "epoch invalid"; return resp;
    }
    CanonicalObject* obj = find_object(req.object_id);
    if (obj && obj->restore_refcount > 0) obj->restore_refcount--;
    // If every holder released while this restore was in flight, reclaim now.
    if (obj) maybe_free_object(*obj);
    resp.ok = true;
    resp.message = "ok";
    return resp;
}

// ---------------------------------------------------------------------------
// DropCanonicalVersion: explicit GC of every object of (job, role, version).
// Keeps host/NVMe bounded when model_version bumps each offload cycle. Skips
// objects with an in-flight export/restore (reported for retry); ignores the
// advisory attachment refcount. Also drops the version's manifest and clears
// the latest-sealed pointer if it referenced this version.
// ---------------------------------------------------------------------------
DropCanonicalVersionResponse OffloadDaemon::handle_drop_canonical_version(
    const DropCanonicalVersionRequest& req) {
    std::lock_guard<std::mutex> lk(mu_);
    DropCanonicalVersionResponse resp;
    if (!cfg_.v2_enable_canonical_store) {
        resp.ok = false; resp.message = "canonical store disabled"; return resp;
    }
    JobRecord* job = find_job_by_wire(req.job);
    if (!job) { resp.ok = false; resp.message = "job not registered"; return resp; }
    const JobUID uid = job->uid;

    // Collect matching object_ids first (release_object mutates objects_).
    std::vector<uint64_t> ids;
    ids.reserve(objects_.size());
    for (auto& kv : objects_) {
        const CanonicalObject& o = kv.second;
        if (o.key.job.tenant_id == uid.tenant_id && o.key.job.job_id == uid.job_id &&
            o.key.job.launch_epoch == uid.launch_epoch &&
            o.key.model_role == req.model_role &&
            o.key.model_version == req.model_version) {
            ids.push_back(o.object_id);
        }
    }
    for (uint64_t oid : ids) {
        CanonicalObject* o = find_object(oid);
        if (!o) continue;
        uint64_t nb = o->nbytes;
        if (release_object(*o)) {
            resp.dropped_count++;
            resp.bytes_freed += nb;
        } else {
            resp.skipped_inflight++;
        }
    }

    // Drop the manifest + latest-sealed pointer for this version.
    VersionKey vk{uid, req.model_role, req.model_version};
    manifests_.erase(vk);
    RoleKey rk{uid, req.model_role};
    auto lsit = latest_sealed_.find(rk);
    if (lsit != latest_sealed_.end() && lsit->second == req.model_version)
        latest_sealed_.erase(lsit);

    resp.ok = true;
    resp.message = "ok";
    OFLD_INFO(TAG, "DropCanonicalVersion job=%llu role=%u version=%llu -> dropped=%llu "
              "skipped=%llu bytes=%llu",
              (unsigned long long)uid.job_id, req.model_role,
              (unsigned long long)req.model_version,
              (unsigned long long)resp.dropped_count,
              (unsigned long long)resp.skipped_inflight,
              (unsigned long long)resp.bytes_freed);
    return resp;
}

// Drop sealed versions of (job, role) beyond retain_sealed_versions, oldest
// first (mu_ held). 0 == unlimited/disabled. Skips the promoted latest and any
// version with an in-flight transfer.
void OffloadDaemon::enforce_sealed_retention(const JobUID& uid, uint32_t model_role) {
    const uint32_t keep = cfg_.v2_retain_sealed_versions;
    if (keep == 0) return;
    // Gather sealed versions for (job, role), newest first.
    std::vector<uint64_t> sealed;
    for (auto& kv : manifests_) {
        const VersionKey& k = kv.first;
        if (k.job == uid && k.model_role == model_role &&
            kv.second.state == MANIFEST_SEALED)
            sealed.push_back(k.model_version);
    }
    if (sealed.size() <= keep) return;
    std::sort(sealed.begin(), sealed.end(), std::greater<uint64_t>());
    RoleKey rk{uid, model_role};
    uint64_t latest = 0;
    auto lsit = latest_sealed_.find(rk);
    if (lsit != latest_sealed_.end()) latest = lsit->second;
    for (size_t i = keep; i < sealed.size(); ++i) {
        uint64_t ver = sealed[i];
        if (ver == latest) continue;   // never evict the promoted pointer
        std::vector<uint64_t> ids;
        for (auto& kv : objects_) {
            const CanonicalObject& o = kv.second;
            if (o.key.job.tenant_id == uid.tenant_id &&
                o.key.job.job_id == uid.job_id &&
                o.key.job.launch_epoch == uid.launch_epoch &&
                o.key.model_role == model_role &&
                o.key.model_version == ver)
                ids.push_back(o.object_id);
        }
        bool all_freed = true;
        for (uint64_t oid : ids) {
            CanonicalObject* o = find_object(oid);
            if (o && !release_object(*o)) all_freed = false;
        }
        if (all_freed) manifests_.erase(VersionKey{uid, model_role, ver});
    }
}

}  // namespace offload
