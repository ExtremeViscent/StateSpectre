// Unit test: config parser (YAML subset) + smoke config defaults.
#include <cstdio>
#include <fstream>
#include <string>

#include "config.h"
#include "test_harness.h"

using namespace offload;
using namespace ofldtest;

static std::string write_temp(const std::string& contents) {
    std::string path = "/tmp/ofld_test_config.yaml";
    std::ofstream f(path);
    f << contents;
    f.close();
    return path;
}

static void test_full_config() {
    // Mirror config/config.example.yaml closely.
    std::string yaml =
        "daemon:\n"
        "  socket_path: /tmp/fastoffload.sock\n"
        "  control_shm_name: /fastoffload_ctrl\n"
        "  heartbeat_timeout_ms: 5000\n"
        "\n"
        "arenas:\n"
        "  registration_policy: lazy_chunked\n"
        "  registration_chunk_mb: 512\n"
        "  allocation_granularity_mb: 16\n"
        "  per_numa:\n"
        "    - numa_node: 0\n"
        "      size_gb: 384\n"
        "      gpu_windows:\n"
        "        - gpu: 0\n"
        "          size_gb: 80\n"
        "        - gpu: 1\n"
        "          size_gb: 80\n"
        "      overflow_gb: 64\n"
        "    - numa_node: 1\n"
        "      size_gb: 384\n"
        "      gpu_windows:\n"
        "        - gpu: 4\n"
        "          size_gb: 80\n"
        "      overflow_gb: 64\n"
        "\n"
        "drain_policy:\n"
        "  target_tier: pageable_then_nvme\n"
        "  drain_on_d2h_complete: true\n"
        "  normal_threshold: 0.60\n"
        "  priority_threshold: 0.80\n"
        "  aggressive_threshold: 0.95\n"
        "\n"
        "nvme:\n"
        "  enabled: true\n"
        "  path: /nvme/fastoffload\n"
        "  io_engine: io_uring\n"
        "  direct_io: true\n"
        "  stripe: true\n";
    auto path = write_temp(yaml);
    DaemonConfig c = load_config(path);

    CHECK_STREQ(c.socket_path, "/tmp/fastoffload.sock");
    CHECK_STREQ(c.control_shm_name, "/fastoffload_ctrl");
    CHECK_EQ(c.heartbeat_timeout_ms, 5000ull);
    CHECK_EQ(c.registration_chunk_bytes, (512ull << 20));
    CHECK_EQ(c.allocation_granularity_bytes, (16ull << 20));

    CHECK_EQ((int)c.per_numa.size(), 2);
    CHECK_EQ(c.per_numa[0].numa_node, 0u);
    CHECK_EQ(c.per_numa[0].size_bytes, (384ull << 30));
    CHECK_EQ((int)c.per_numa[0].gpu_windows.size(), 2);
    CHECK_EQ(c.per_numa[0].gpu_windows[0].gpu, 0u);
    CHECK_EQ(c.per_numa[0].gpu_windows[0].size_bytes, (80ull << 30));
    CHECK_EQ(c.per_numa[0].gpu_windows[1].gpu, 1u);
    CHECK_EQ(c.per_numa[0].overflow_bytes, (64ull << 30));
    CHECK_EQ(c.per_numa[1].numa_node, 1u);
    CHECK_EQ(c.per_numa[1].gpu_windows[0].gpu, 4u);

    CHECK(c.drain_target == DrainTarget::kPageableThenNvme);
    CHECK(c.drain_on_d2h_complete);
    CHECK(c.nvme_enabled);
    CHECK_STREQ(c.nvme_path, "/nvme/fastoffload");
    CHECK(c.nvme_direct_io);
    CHECK(c.nvme_stripe);
}

static void test_defaults_applied() {
    // Minimal valid config: a socket path + one arena. Everything else defaults.
    std::string yaml =
        "daemon:\n"
        "  socket_path: /tmp/x.sock\n"
        "arenas:\n"
        "  per_numa:\n"
        "    - numa_node: 0\n"
        "      size_gb: 1\n"
        "      gpu_windows:\n"
        "        - gpu: 0\n"
        "          size_gb: 1\n"
        "      overflow_gb: 0\n";
    auto path = write_temp(yaml);
    DaemonConfig c = load_config(path);
    CHECK_STREQ(c.socket_path, "/tmp/x.sock");
    // Defaults from the struct:
    CHECK_EQ(c.heartbeat_timeout_ms, 5000ull);
    CHECK(c.drain_on_d2h_complete);
    CHECK_EQ((int)c.per_numa.size(), 1);
}

static void test_smoke_config() {
    DaemonConfig c = default_smoke_config(256ull << 20, 0, 0);
    CHECK_EQ((int)c.per_numa.size(), 1);
    CHECK_EQ(c.per_numa[0].numa_node, 0u);
    CHECK_EQ((int)c.per_numa[0].gpu_windows.size(), 1);
    CHECK_EQ(c.per_numa[0].gpu_windows[0].gpu, 0u);
    // The arena is carved into a GPU window + an overflow window, so the window
    // is the arena minus overflow. Total must not exceed the arena size.
    uint64_t win = c.per_numa[0].gpu_windows[0].size_bytes;
    uint64_t ovf = c.per_numa[0].overflow_bytes;
    CHECK(win > 0);
    CHECK(win + ovf <= (256ull << 20));
    CHECK(win + ovf >= (256ull << 20) - c.allocation_granularity_bytes);
    CHECK(!c.nvme_enabled);
    // allocation granularity should be small for tests.
    CHECK(c.allocation_granularity_bytes <= (2ull << 20));
}

static void test_missing_file_throws() {
    bool threw = false;
    try {
        load_config("/tmp/definitely_does_not_exist_ofld.yaml");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_tab_rejected() {
    // Tabs are illegal indentation in YAML; parser should reject.
    std::string yaml = "daemon:\n\tsocket_path: /tmp/x.sock\n";
    auto path = write_temp(yaml);
    bool threw = false;
    try {
        load_config(path);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

// v2 canonical/quota/export overlay keys parse into DaemonConfig, including IEC
// byte suffixes and enum name -> numeric mapping.
static void test_v2_config() {
    std::string yaml =
        "daemon:\n"
        "  socket_path: /tmp/fastoffload.sock\n"
        "arenas:\n"
        "  allocation_granularity_mb: 16\n"
        "  per_numa:\n"
        "    - numa_node: 0\n"
        "      size_gb: 8\n"
        "      gpu_windows:\n"
        "        - gpu: 0\n"
        "          size_gb: 4\n"
        "      overflow_gb: 2\n"
        "manager:\n"
        "  enable_canonical_store: true\n"
        "  control_tcp_port: 19090\n"
        "namespace:\n"
        "  default_tenant_id: 7\n"
        "  include_launch_epoch: true\n"
        "canonical_store:\n"
        "  dedup_default_mode: hash_verified\n"
        "  creating_policy: duplicate_candidate_on_pressure\n"
        "  duplicate_candidate_gpu_pressure_threshold: 0.75\n"
        "manifest:\n"
        "  retain_sealed_versions: 8\n"
        "export:\n"
        "  default_transport: libfabric_send\n"
        "  fallback_transport: tcp\n"
        "  max_concurrent_exports_per_job: 16\n"
        "quotas:\n"
        "  default:\n"
        "    max_pinned_bytes: 256GiB\n"
        "    max_nvme_bytes: 8TiB\n"
        "    max_inflight_d2h_bytes: 160GiB\n";
    auto path = write_temp(yaml);
    DaemonConfig c = load_config(path);
    CHECK(c.v2_enable_canonical_store);
    CHECK_EQ(c.v2_control_tcp_port, 19090u);
    CHECK_EQ(c.v2_default_tenant_id, 7ull);
    CHECK_EQ(c.v2_dedup_default_mode, 2u);       // hash_verified
    CHECK_EQ(c.v2_creating_policy, 1u);          // duplicate_candidate_on_pressure
    CHECK_EQ(c.v2_retain_sealed_versions, 8u);
    CHECK_EQ(c.v2_default_transport, 3u);        // libfabric_send
    CHECK_EQ(c.v2_fallback_transport, 1u);       // tcp
    CHECK_EQ(c.v2_max_concurrent_exports_per_job, 16u);
    CHECK_EQ(c.v2_quota_max_pinned_bytes, (256ull << 30));
    CHECK_EQ(c.v2_quota_max_nvme_bytes, (8ull << 40));
    CHECK_EQ(c.v2_quota_max_inflight_d2h_bytes, (160ull << 30));
    CHECK(c.v2_duplicate_candidate_gpu_pressure_threshold > 0.74 &&
          c.v2_duplicate_candidate_gpu_pressure_threshold < 0.76);
}

int main() {
    RUN(test_full_config);
    RUN(test_defaults_applied);
    RUN(test_smoke_config);
    RUN(test_missing_file_throws);
    RUN(test_tab_rejected);
    RUN(test_v2_config);
    return summary("config");
}
