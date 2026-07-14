// Configuration model + hand-rolled YAML-subset parser for the daemon.
//
// Implements everything declared in config.h. We parse the YAML subset shown in
// config/config.example.yaml directly (indentation-based "key: value" maps and
// "- key: value" lists of maps) without any YAML library dependency.

#include "config.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "util.h"

namespace offload {

namespace {

constexpr uint64_t kMiB = 1ull << 20;
constexpr uint64_t kGiB = 1ull << 30;

// ------------------------------------------------------------------
// Minimal YAML-subset tree.
// ------------------------------------------------------------------
struct YamlNode {
    enum Kind { kEmpty, kScalar, kMap, kSeq } kind = kEmpty;
    std::string scalar;
    std::map<std::string, YamlNode> map;
    std::vector<YamlNode> seq;
    int line = 0;  // source line (1-based) where this node began

    const YamlNode* find(const std::string& key) const {
        auto it = map.find(key);
        return it == map.end() ? nullptr : &it->second;
    }
};

struct SrcLine {
    int indent;
    int lineno;       // 1-based original line number
    std::string text; // content with leading indentation stripped
};

std::string rstrip(const std::string& s) {
    size_t e = s.size();
    while (e > 0 && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r'))
        --e;
    return s.substr(0, e);
}

std::string strip(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r')) ++b;
    size_t e = s.size();
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r')) --e;
    return s.substr(b, e - b);
}

// Strip a trailing "# ..." comment. Our config values never contain '#'.
std::string strip_comment(const std::string& s) {
    size_t pos = s.find('#');
    if (pos == std::string::npos) return s;
    return s.substr(0, pos);
}

// Tokenize the file into non-blank logical lines with computed indentation.
std::vector<SrcLine> tokenize(const std::string& content) {
    std::vector<SrcLine> out;
    std::string line;
    size_t i = 0;
    int lineno = 0;
    while (i <= content.size()) {
        if (i == content.size() || content[i] == '\n') {
            ++lineno;
            std::string raw = line;
            line.clear();
            std::string noc = strip_comment(raw);
            std::string trimmed = rstrip(noc);
            std::string only = strip(trimmed);
            if (!only.empty()) {
                int indent = 0;
                while (indent < static_cast<int>(trimmed.size()) &&
                       trimmed[indent] == ' ')
                    ++indent;
                if (trimmed.find('\t') != std::string::npos &&
                    static_cast<size_t>(trimmed.find_first_not_of(" \t")) >
                        static_cast<size_t>(indent)) {
                    throw std::runtime_error(
                        "config parse error at line " + std::to_string(lineno) +
                        ": tabs are not allowed for indentation");
                }
                SrcLine sl;
                sl.indent = indent;
                sl.lineno = lineno;
                sl.text = trimmed.substr(indent);
                out.push_back(std::move(sl));
            }
            if (i == content.size()) break;
        } else {
            line.push_back(content[i]);
        }
        ++i;
    }
    return out;
}

// Forward decl.
YamlNode parse_node(std::vector<SrcLine>& lines, size_t& i, int indent);

YamlNode parse_map(std::vector<SrcLine>& lines, size_t& i, int indent) {
    YamlNode node;
    node.kind = YamlNode::kMap;
    node.line = i < lines.size() ? lines[i].lineno : 0;
    while (i < lines.size() && lines[i].indent == indent &&
           lines[i].text.rfind("- ", 0) != 0 && lines[i].text != "-") {
        const std::string& t = lines[i].text;
        size_t colon = t.find(':');
        if (colon == std::string::npos) {
            throw std::runtime_error("config parse error at line " +
                                     std::to_string(lines[i].lineno) +
                                     ": expected 'key: value'");
        }
        std::string key = strip(t.substr(0, colon));
        std::string val = strip(t.substr(colon + 1));
        int cur_line = lines[i].lineno;
        ++i;
        if (!val.empty()) {
            YamlNode child;
            child.kind = YamlNode::kScalar;
            child.scalar = val;
            child.line = cur_line;
            node.map[key] = std::move(child);
        } else {
            // Nested block if the next line is more deeply indented.
            if (i < lines.size() && lines[i].indent > indent) {
                YamlNode child = parse_node(lines, i, lines[i].indent);
                node.map[key] = std::move(child);
            } else {
                YamlNode child;
                child.kind = YamlNode::kEmpty;
                child.line = cur_line;
                node.map[key] = std::move(child);
            }
        }
    }
    return node;
}

YamlNode parse_seq(std::vector<SrcLine>& lines, size_t& i, int indent) {
    YamlNode node;
    node.kind = YamlNode::kSeq;
    node.line = i < lines.size() ? lines[i].lineno : 0;
    while (i < lines.size() && lines[i].indent == indent &&
           (lines[i].text.rfind("- ", 0) == 0 || lines[i].text == "-")) {
        const std::string& t = lines[i].text;
        // Find where item content begins (after '-' and following spaces).
        size_t c = 1;  // skip '-'
        while (c < t.size() && t[c] == ' ') ++c;
        std::string content = t.substr(c);
        int item_indent = indent + static_cast<int>(c);
        int cur_line = lines[i].lineno;
        if (content.empty()) {
            // Nested block belongs on subsequent, more-indented lines.
            ++i;
            if (i < lines.size() && lines[i].indent > indent) {
                YamlNode child = parse_node(lines, i, lines[i].indent);
                node.seq.push_back(std::move(child));
            } else {
                YamlNode child;
                child.kind = YamlNode::kEmpty;
                child.line = cur_line;
                node.seq.push_back(std::move(child));
            }
        } else {
            // Rewrite this line as if the item content started at item_indent,
            // then parse a node at that indent (map / scalar spanning the item).
            lines[i].indent = item_indent;
            lines[i].text = content;
            YamlNode child = parse_node(lines, i, item_indent);
            node.seq.push_back(std::move(child));
        }
    }
    return node;
}

YamlNode parse_node(std::vector<SrcLine>& lines, size_t& i, int indent) {
    if (i >= lines.size()) {
        YamlNode n;
        n.kind = YamlNode::kEmpty;
        return n;
    }
    const std::string& t = lines[i].text;
    if (t.rfind("- ", 0) == 0 || t == "-") {
        return parse_seq(lines, i, indent);
    }
    // A scalar item (e.g. inside a plain sequence) has no ':'.
    if (t.find(':') == std::string::npos) {
        YamlNode n;
        n.kind = YamlNode::kScalar;
        n.scalar = strip(t);
        n.line = lines[i].lineno;
        ++i;
        return n;
    }
    return parse_map(lines, i, indent);
}

// ------------------------------------------------------------------
// Scalar conversions.
// ------------------------------------------------------------------
uint64_t to_u64(const YamlNode& n, const std::string& ctx) {
    if (n.kind != YamlNode::kScalar)
        throw std::runtime_error("config error at line " + std::to_string(n.line) +
                                 ": expected integer for " + ctx);
    try {
        size_t pos = 0;
        long long v = std::stoll(n.scalar, &pos);
        if (pos != n.scalar.size() || v < 0)
            throw std::invalid_argument("bad");
        return static_cast<uint64_t>(v);
    } catch (const std::exception&) {
        throw std::runtime_error("config error at line " + std::to_string(n.line) +
                                 ": invalid integer '" + n.scalar + "' for " + ctx);
    }
}

double to_double(const YamlNode& n, const std::string& ctx) {
    if (n.kind != YamlNode::kScalar)
        throw std::runtime_error("config error at line " + std::to_string(n.line) +
                                 ": expected number for " + ctx);
    try {
        size_t pos = 0;
        double v = std::stod(n.scalar, &pos);
        if (pos != n.scalar.size()) throw std::invalid_argument("bad");
        return v;
    } catch (const std::exception&) {
        throw std::runtime_error("config error at line " + std::to_string(n.line) +
                                 ": invalid number '" + n.scalar + "' for " + ctx);
    }
}

bool to_bool(const YamlNode& n, const std::string& ctx) {
    if (n.kind != YamlNode::kScalar)
        throw std::runtime_error("config error at line " + std::to_string(n.line) +
                                 ": expected bool for " + ctx);
    std::string s = n.scalar;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (s == "true" || s == "yes" || s == "on" || s == "1") return true;
    if (s == "false" || s == "no" || s == "off" || s == "0") return false;
    throw std::runtime_error("config error at line " + std::to_string(n.line) +
                             ": invalid bool '" + n.scalar + "' for " + ctx);
}

std::string to_str(const YamlNode& n, const std::string& ctx) {
    if (n.kind != YamlNode::kScalar)
        throw std::runtime_error("config error at line " + std::to_string(n.line) +
                                 ": expected string for " + ctx);
    return n.scalar;
}

DrainTarget parse_target(const std::string& s, int line) {
    if (s == "pageable_then_nvme") return DrainTarget::kPageableThenNvme;
    if (s == "pageable") return DrainTarget::kPageableOnly;
    if (s == "nvme") return DrainTarget::kNvmeOnly;
    throw std::runtime_error("config error at line " + std::to_string(line) +
                             ": unknown target_tier '" + s + "'");
}

// Parse a byte-size scalar with an optional binary/decimal suffix, e.g.
// "256GiB", "1TiB", "160GiB", "512MiB", or a plain integer (bytes). Used by the
// v2 quota/export keys which are written with IEC suffixes.
uint64_t to_bytes(const YamlNode& n, const std::string& ctx) {
    if (n.kind != YamlNode::kScalar)
        throw std::runtime_error("config error at line " + std::to_string(n.line) +
                                 ": expected size for " + ctx);
    const std::string& s = n.scalar;
    size_t pos = 0;
    double v = 0;
    try { v = std::stod(s, &pos); }
    catch (const std::exception&) {
        throw std::runtime_error("config error at line " + std::to_string(n.line) +
                                 ": invalid size '" + s + "' for " + ctx);
    }
    std::string suf = s.substr(pos);
    // trim spaces
    while (!suf.empty() && suf.front() == ' ') suf.erase(suf.begin());
    while (!suf.empty() && suf.back() == ' ') suf.pop_back();
    std::transform(suf.begin(), suf.end(), suf.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    uint64_t mult = 1;
    if (suf.empty() || suf == "b") mult = 1;
    else if (suf == "kib" || suf == "kb" || suf == "k") mult = 1ull << 10;
    else if (suf == "mib" || suf == "mb" || suf == "m") mult = 1ull << 20;
    else if (suf == "gib" || suf == "gb" || suf == "g") mult = 1ull << 30;
    else if (suf == "tib" || suf == "tb" || suf == "t") mult = 1ull << 40;
    else
        throw std::runtime_error("config error at line " + std::to_string(n.line) +
                                 ": unknown size suffix '" + suf + "' for " + ctx);
    return static_cast<uint64_t>(v * static_cast<double>(mult));
}

// Map a dedup mode name to the DedupMode numeric (offload_canonical_abi.hpp).
uint32_t parse_dedup_mode(const std::string& s) {
    if (s == "disabled") return 0;
    if (s == "semantic_trusted") return 1;
    if (s == "hash_verified") return 2;
    if (s == "sampled_hash") return 3;
    return 1;  // default semantic_trusted
}

// Map a transport name to TransportKind numeric.
uint32_t parse_transport(const std::string& s) {
    if (s == "tcp") return 1;
    if (s == "file") return 2;
    if (s == "libfabric_send" || s == "libfabric") return 3;
    if (s == "libfabric_rdma_read") return 4;
    if (s == "libfabric_rdma_write") return 5;
    return 1;
}

}  // namespace

// ----------------------------------------------------------------------------

const char* drain_target_name(DrainTarget t) {
    switch (t) {
        case DrainTarget::kPageableOnly:     return "pageable";
        case DrainTarget::kNvmeOnly:         return "nvme";
        case DrainTarget::kPageableThenNvme: return "pageable_then_nvme";
    }
    return "unknown";
}

DaemonConfig default_smoke_config(uint64_t arena_bytes, uint32_t numa_node,
                                  uint32_t gpu_id) {
    DaemonConfig cfg;
    cfg.allocation_granularity_bytes = 2ull * kMiB;   // small so tests use small arenas
    cfg.registration_chunk_bytes =
        std::min<uint64_t>(arena_bytes, 512ull * kMiB);
    cfg.nvme_enabled = false;
    cfg.drain_workers = 1;
    cfg.nvme_workers = 1;
    cfg.drain_target = DrainTarget::kPageableOnly;  // no NVMe in smoke

    NumaArenaConfig arena;
    arena.numa_node = numa_node;
    arena.size_bytes = arena_bytes;

    GpuWindowConfig win;
    win.gpu = gpu_id;
    // Single GPU window covers the whole arena; overflow carved out below.
    uint64_t overflow = std::max<uint64_t>(arena_bytes / 8, 16ull * kMiB);
    if (overflow >= arena_bytes) overflow = arena_bytes / 2;
    win.size_bytes = arena_bytes - overflow;
    arena.gpu_windows.push_back(win);
    arena.overflow_bytes = overflow;

    cfg.per_numa.push_back(std::move(arena));
    return cfg;
}

DaemonConfig load_config(const std::string& path) {
    if (path.empty()) {
        // No file: minimal single-NUMA smoke config.
        return default_smoke_config(256ull * kMiB, 0, 0);
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("config: cannot open file '" + path + "'");
    }
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());

    std::vector<SrcLine> lines = tokenize(content);
    if (lines.empty()) {
        throw std::runtime_error("config: file '" + path + "' is empty");
    }
    size_t i = 0;
    YamlNode root = parse_node(lines, i, lines[0].indent);
    if (i != lines.size()) {
        throw std::runtime_error("config parse error at line " +
                                 std::to_string(lines[i].lineno) +
                                 ": unexpected content / bad indentation");
    }
    if (root.kind != YamlNode::kMap) {
        throw std::runtime_error("config: top-level must be a mapping");
    }

    DaemonConfig cfg;

    // Known top-level sections (v1 + v2 canonical overlay).
    static const char* kTopKeys[] = {"daemon", "arenas", "drain_policy", "nvme",
                                      "python_api", "manager", "namespace",
                                      "canonical_store", "manifest", "export",
                                      "quotas", "roles", "libfabric", "cleanup",
                                      "metrics"};
    for (const auto& kv : root.map) {
        bool known = false;
        for (const char* k : kTopKeys)
            if (kv.first == k) known = true;
        if (!known)
            OFLD_WARN("config", "unknown top-level key '%s' (ignored)",
                      kv.first.c_str());
    }

    // ---- daemon: ----
    if (const YamlNode* d = root.find("daemon")) {
        for (const auto& kv : d->map) {
            const std::string& k = kv.first;
            const YamlNode& v = kv.second;
            if (k == "socket_path") cfg.socket_path = to_str(v, k);
            else if (k == "control_shm_name") cfg.control_shm_name = to_str(v, k);
            else if (k == "heartbeat_timeout_ms")
                cfg.heartbeat_timeout_ms = to_u64(v, k);
            else if (k == "completion_ring_max_ranks")
                cfg.completion_ring_max_ranks = static_cast<uint32_t>(to_u64(v, k));
            else if (k == "completion_ring_capacity")
                cfg.completion_ring_capacity = static_cast<uint32_t>(to_u64(v, k));
            else if (k == "ring_poll_us")
                cfg.ring_poll_us = static_cast<uint32_t>(to_u64(v, k));
            else OFLD_WARN("config", "unknown daemon.%s (ignored)", k.c_str());
        }
    }

    // ---- arenas: ----
    if (const YamlNode* a = root.find("arenas")) {
        for (const auto& kv : a->map) {
            const std::string& k = kv.first;
            const YamlNode& v = kv.second;
            if (k == "registration_policy") cfg.registration_policy = to_str(v, k);
            else if (k == "registration_chunk_mb")
                cfg.registration_chunk_bytes = to_u64(v, k) * kMiB;
            else if (k == "allocation_granularity_mb")
                cfg.allocation_granularity_bytes = to_u64(v, k) * kMiB;
            else if (k == "per_numa") { /* handled below */ }
            else OFLD_WARN("config", "unknown arenas.%s (ignored)", k.c_str());
        }
        if (const YamlNode* pn = a->find("per_numa")) {
            if (pn->kind != YamlNode::kSeq)
                throw std::runtime_error("config error at line " +
                                         std::to_string(pn->line) +
                                         ": arenas.per_numa must be a list");
            for (const YamlNode& item : pn->seq) {
                if (item.kind != YamlNode::kMap)
                    throw std::runtime_error(
                        "config error at line " + std::to_string(item.line) +
                        ": per_numa entry must be a mapping");
                NumaArenaConfig arena;
                for (const auto& akv : item.map) {
                    const std::string& k = akv.first;
                    const YamlNode& v = akv.second;
                    if (k == "numa_node")
                        arena.numa_node = static_cast<uint32_t>(to_u64(v, k));
                    else if (k == "size_gb")
                        arena.size_bytes = to_u64(v, k) * kGiB;
                    else if (k == "overflow_gb")
                        arena.overflow_bytes = to_u64(v, k) * kGiB;
                    else if (k == "gpu_windows") { /* below */ }
                    else
                        OFLD_WARN("config", "unknown per_numa.%s (ignored)",
                                  k.c_str());
                }
                if (const YamlNode* gw = item.find("gpu_windows")) {
                    if (gw->kind != YamlNode::kSeq)
                        throw std::runtime_error(
                            "config error at line " + std::to_string(gw->line) +
                            ": gpu_windows must be a list");
                    for (const YamlNode& w : gw->seq) {
                        if (w.kind != YamlNode::kMap)
                            throw std::runtime_error(
                                "config error at line " + std::to_string(w.line) +
                                ": gpu_windows entry must be a mapping");
                        GpuWindowConfig win;
                        for (const auto& wkv : w.map) {
                            const std::string& k = wkv.first;
                            const YamlNode& v = wkv.second;
                            if (k == "gpu")
                                win.gpu = static_cast<uint32_t>(to_u64(v, k));
                            else if (k == "size_gb")
                                win.size_bytes = to_u64(v, k) * kGiB;
                            else
                                OFLD_WARN("config",
                                          "unknown gpu_windows.%s (ignored)",
                                          k.c_str());
                        }
                        arena.gpu_windows.push_back(win);
                    }
                }
                cfg.per_numa.push_back(std::move(arena));
            }
        }
    }

    // ---- drain_policy: ----
    if (const YamlNode* dp = root.find("drain_policy")) {
        for (const auto& kv : dp->map) {
            const std::string& k = kv.first;
            const YamlNode& v = kv.second;
            if (k == "target_tier")
                cfg.drain_target = parse_target(to_str(v, k), v.line);
            else if (k == "drain_on_d2h_complete")
                cfg.drain_on_d2h_complete = to_bool(v, k);
            else if (k == "normal_threshold")
                cfg.normal_threshold = to_double(v, k);
            else if (k == "priority_threshold")
                cfg.priority_threshold = to_double(v, k);
            else if (k == "aggressive_threshold")
                cfg.aggressive_threshold = to_double(v, k);
            else
                OFLD_WARN("config", "unknown drain_policy.%s (ignored)", k.c_str());
        }
    }

    // ---- nvme: ----
    if (const YamlNode* nv = root.find("nvme")) {
        for (const auto& kv : nv->map) {
            const std::string& k = kv.first;
            const YamlNode& v = kv.second;
            if (k == "enabled") cfg.nvme_enabled = to_bool(v, k);
            else if (k == "path") cfg.nvme_path = to_str(v, k);
            else if (k == "io_engine") cfg.nvme_io_engine = to_str(v, k);
            else if (k == "direct_io") cfg.nvme_direct_io = to_bool(v, k);
            else if (k == "stripe") cfg.nvme_stripe = to_bool(v, k);
            else if (k == "stripe_count")
                cfg.nvme_stripe_count = static_cast<uint32_t>(to_u64(v, k));
            else if (k == "block_mb")
                cfg.nvme_block_bytes = to_u64(v, k) * 1024ull * 1024ull;
            else if (k == "queue_depth")
                cfg.nvme_queue_depth = static_cast<uint32_t>(to_u64(v, k));
            else if (k == "stripe_dirs") {
                cfg.nvme_stripe_dirs.clear();
                for (const auto& item : v.seq)
                    cfg.nvme_stripe_dirs.push_back(item.scalar);
            }
            else OFLD_WARN("config", "unknown nvme.%s (ignored)", k.c_str());
        }
    }

    // ---- python_api: (informational, enforced rank-side) ----
    if (const YamlNode* pa = root.find("python_api")) {
        for (const auto& kv : pa->map) {
            const std::string& k = kv.first;
            const YamlNode& v = kv.second;
            if (k == "default_invalidate") cfg.default_invalidate = to_str(v, k);
            else if (k == "require_own_storage")
                cfg.require_own_storage = to_bool(v, k);
            else if (k == "allow_views") cfg.allow_views = to_bool(v, k);
            else if (k == "unsafe_autograd") cfg.unsafe_autograd = to_bool(v, k);
            else OFLD_WARN("config", "unknown python_api.%s (ignored)", k.c_str());
        }
    }

    // ---- v2: manager: (feature toggles + optional TCP control port) ----
    if (const YamlNode* mg = root.find("manager")) {
        for (const auto& kv : mg->map) {
            const std::string& k = kv.first;
            const YamlNode& v = kv.second;
            if (k == "enable_canonical_store")
                cfg.v2_enable_canonical_store = to_bool(v, k);
            else if (k == "enable_manifests")
                cfg.v2_enable_manifests = to_bool(v, k);
            else if (k == "enable_rollout_export")
                cfg.v2_enable_rollout_export = to_bool(v, k);
            else if (k == "control_tcp_port")
                cfg.v2_control_tcp_port = static_cast<uint32_t>(to_u64(v, k));
            else if (k == "daemon_socket") { /* alias of daemon.socket_path */
                cfg.socket_path = to_str(v, k);
            }
            else OFLD_WARN("config", "unknown manager.%s (ignored)", k.c_str());
        }
    }

    // ---- v2: namespace: ----
    if (const YamlNode* ns = root.find("namespace")) {
        for (const auto& kv : ns->map) {
            const std::string& k = kv.first;
            const YamlNode& v = kv.second;
            if (k == "default_tenant_id")
                cfg.v2_default_tenant_id = to_u64(v, k);
            else if (k == "require_job_registration")
                cfg.v2_require_job_registration = to_bool(v, k);
            else if (k == "include_launch_epoch")
                cfg.v2_include_launch_epoch = to_bool(v, k);
            else if (k == "job_name_is_identity") { /* enforced false; informational */ }
            else OFLD_WARN("config", "unknown namespace.%s (ignored)", k.c_str());
        }
    }

    // ---- v2: canonical_store: ----
    if (const YamlNode* cs = root.find("canonical_store")) {
        for (const auto& kv : cs->map) {
            const std::string& k = kv.first;
            const YamlNode& v = kv.second;
            if (k == "dedup_default_mode")
                cfg.v2_dedup_default_mode = parse_dedup_mode(to_str(v, k));
            else if (k == "cross_job_dedup")
                cfg.v2_cross_job_dedup = to_bool(v, k);
            else if (k == "creating_policy")
                cfg.v2_creating_policy =
                    (to_str(v, k) == "duplicate_candidate_on_pressure") ? 1u : 0u;
            else if (k == "duplicate_candidate_gpu_pressure_threshold")
                cfg.v2_duplicate_candidate_gpu_pressure_threshold = to_double(v, k);
            else if (k == "sampled_hash_bytes_per_gib")
                cfg.v2_sampled_hash_bytes_per_gib = to_u64(v, k);
            else if (k == "object_table_capacity" || k == "manifest_table_capacity") {
                /* informational: daemon tables grow dynamically */
            }
            else OFLD_WARN("config", "unknown canonical_store.%s (ignored)", k.c_str());
        }
    }

    // ---- v2: manifest: ----
    if (const YamlNode* mf = root.find("manifest")) {
        for (const auto& kv : mf->map) {
            const std::string& k = kv.first;
            const YamlNode& v = kv.second;
            if (k == "seal_requires_all_objects_valid")
                cfg.v2_seal_requires_all_objects_valid = to_bool(v, k);
            else if (k == "retain_sealed_versions")
                cfg.v2_retain_sealed_versions = static_cast<uint32_t>(to_u64(v, k));
            else if (k == "allow_sealed_version_mutation" ||
                     k == "latest_pointer_update") { /* informational */ }
            else OFLD_WARN("config", "unknown manifest.%s (ignored)", k.c_str());
        }
    }

    // ---- v2: export: ----
    if (const YamlNode* ex = root.find("export")) {
        for (const auto& kv : ex->map) {
            const std::string& k = kv.first;
            const YamlNode& v = kv.second;
            if (k == "default_transport")
                cfg.v2_default_transport = parse_transport(to_str(v, k));
            else if (k == "fallback_transport")
                cfg.v2_fallback_transport = parse_transport(to_str(v, k));
            else if (k == "use_dedicated_export_buffers")
                cfg.v2_use_dedicated_export_buffers = to_bool(v, k);
            else if (k == "require_export_refcount")
                cfg.v2_require_export_refcount = to_bool(v, k);
            else if (k == "max_concurrent_exports_per_job")
                cfg.v2_max_concurrent_exports_per_job =
                    static_cast<uint32_t>(to_u64(v, k));
            else if (k == "export_buffer_pool_bytes" ||
                     k == "allow_direct_rdma_read_from_object_slot") {
                /* informational for the debug-transport implementation */
            }
            else OFLD_WARN("config", "unknown export.%s (ignored)", k.c_str());
        }
    }

    // ---- v2: quotas.default: (per-job default quotas) ----
    if (const YamlNode* q = root.find("quotas")) {
        if (const YamlNode* def = q->find("default")) {
            for (const auto& kv : def->map) {
                const std::string& k = kv.first;
                const YamlNode& v = kv.second;
                if (k == "max_pinned_bytes")
                    cfg.v2_quota_max_pinned_bytes = to_bytes(v, k);
                else if (k == "max_pageable_bytes")
                    cfg.v2_quota_max_pageable_bytes = to_bytes(v, k);
                else if (k == "max_nvme_bytes")
                    cfg.v2_quota_max_nvme_bytes = to_bytes(v, k);
                else if (k == "max_inflight_d2h_bytes")
                    cfg.v2_quota_max_inflight_d2h_bytes = to_bytes(v, k);
                else if (k == "max_inflight_export_bytes")
                    cfg.v2_quota_max_inflight_export_bytes = to_bytes(v, k);
                else if (k == "priority") { /* informational */ }
                else OFLD_WARN("config", "unknown quotas.default.%s (ignored)",
                               k.c_str());
            }
        }
    }

    // ---- validation ----
    if (cfg.allocation_granularity_bytes == 0)
        throw std::runtime_error("config: allocation_granularity_mb must be > 0");
    if (cfg.per_numa.empty())
        throw std::runtime_error("config: at least one arenas.per_numa entry required");
    for (const auto& arena : cfg.per_numa) {
        uint64_t win_total = 0;
        for (const auto& w : arena.gpu_windows) win_total += w.size_bytes;
        uint64_t needed = win_total + arena.overflow_bytes;
        if (arena.size_bytes == 0)
            throw std::runtime_error("config: numa_node " +
                                     std::to_string(arena.numa_node) +
                                     " size_gb must be > 0");
        if (needed > arena.size_bytes)
            throw std::runtime_error(
                "config: numa_node " + std::to_string(arena.numa_node) +
                " gpu_windows + overflow (" + std::to_string(needed) +
                " bytes) exceed arena size (" + std::to_string(arena.size_bytes) +
                " bytes)");
    }

    return cfg;
}

}  // namespace offload
