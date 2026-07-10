// offloadd: CLI entry point for the centralized GPU tensor-offload daemon.
//
// Usage:
//   offloadd --config PATH            [--socket PATH] [--trace PATH]
//   offloadd --smoke-arena-mb N [--numa N] [--gpu N] [--socket PATH] [--trace PATH]
//
// With no --config and no --smoke-arena-mb, a minimal single-NUMA smoke config
// is used (256 MB arena on node 0, gpu 0).

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "config.h"
#include "daemon.h"
#include "metrics.h"
#include "util.h"

namespace {

offload::OffloadDaemon* g_daemon = nullptr;

void handle_signal(int sig) {
    (void)sig;
    if (g_daemon) g_daemon->request_stop();
}

void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s [--config PATH] [--socket PATH] [--trace PATH]\n"
        "          [--smoke-arena-mb N] [--numa N] [--gpu N] [--tcp-port N]\n",
        argv0);
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path;
    std::string socket_override;
    std::string trace_path;
    uint64_t smoke_arena_mb = 0;
    uint32_t numa = 0;
    uint32_t gpu = 0;
    uint32_t tcp_port = 0;
    bool have_smoke = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s requires an argument\n", name);
                usage(argv[0]);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--config") config_path = next("--config");
        else if (a == "--socket") socket_override = next("--socket");
        else if (a == "--trace") trace_path = next("--trace");
        else if (a == "--smoke-arena-mb") {
            smoke_arena_mb = std::strtoull(next("--smoke-arena-mb").c_str(), nullptr, 10);
            have_smoke = true;
        }
        else if (a == "--numa") numa = static_cast<uint32_t>(std::strtoul(next("--numa").c_str(), nullptr, 10));
        else if (a == "--gpu") gpu = static_cast<uint32_t>(std::strtoul(next("--gpu").c_str(), nullptr, 10));
        else if (a == "--tcp-port") tcp_port = static_cast<uint32_t>(std::strtoul(next("--tcp-port").c_str(), nullptr, 10));
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else {
            std::fprintf(stderr, "error: unknown argument '%s'\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    offload::DaemonConfig cfg;
    try {
        if (!config_path.empty()) {
            cfg = offload::load_config(config_path);
        } else if (have_smoke) {
            cfg = offload::default_smoke_config(smoke_arena_mb << 20, numa, gpu);
        } else {
            OFLD_INFO("main", "no --config or --smoke-arena-mb; using default smoke config");
            cfg = offload::default_smoke_config(256ull << 20, numa, gpu);
        }
    } catch (const std::exception& e) {
        OFLD_ERR("main", "config error: %s", e.what());
        return 1;
    }

    if (!socket_override.empty()) cfg.socket_path = socket_override;
    if (tcp_port != 0) cfg.v2_control_tcp_port = tcp_port;

    if (!trace_path.empty()) {
        offload::Trace::instance().enable(trace_path);
    } else if (const char* env = std::getenv("OFLD_TRACE")) {
        if (env[0]) offload::Trace::instance().enable(env);
    }

    // Ignore SIGPIPE: dead client sockets must not kill the daemon.
    std::signal(SIGPIPE, SIG_IGN);

    try {
        offload::OffloadDaemon daemon(cfg);
        g_daemon = &daemon;

        struct sigaction sa;
        std::memset(&sa, 0, sizeof(sa));
        sa.sa_handler = handle_signal;
        sigaction(SIGINT, &sa, nullptr);
        sigaction(SIGTERM, &sa, nullptr);

        daemon.run();
        g_daemon = nullptr;
    } catch (const std::exception& e) {
        OFLD_ERR("main", "fatal: %s", e.what());
        return 1;
    }

    offload::Trace::instance().disable();
    return 0;
}
