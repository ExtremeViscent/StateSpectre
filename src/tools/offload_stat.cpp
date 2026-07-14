// offload-stat: standalone administrative CLI for the GPU-offload daemon.
//
// Connects to the daemon's Unix domain socket, issues a GetStats RPC, and
// prints all counters plus bandwidth percentiles. It can also emit
// Prometheus text-exposition format, watch continuously, or ask the daemon
// to shut down.
//
// This tool links ONLY against the CUDA-free common transport (wire + uds +
// metrics + numa). It never pulls in libtorch or the CUDA runtime.
//
// Usage:
//   offload-stat [--socket PATH] [--prometheus] [--watch SECONDS] [--shutdown]

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <unistd.h>

#include "protocol.h"
#include "uds.h"
#include "wire.h"

using namespace offload;

namespace {

constexpr const char* kDefaultSocket = "/tmp/state_spectre.sock";

// Set by SIGINT/SIGTERM so the --watch loop can exit cleanly between iterations.
volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

struct Options {
    std::string socket_path = kDefaultSocket;
    bool prometheus = false;
    bool shutdown = false;
    bool watch = false;
    int watch_seconds = 0;
};

void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s [--socket PATH] [--prometheus] [--watch SECONDS] [--shutdown]\n"
        "  --socket PATH    daemon socket (default %s)\n"
        "  --prometheus     emit Prometheus text-exposition format\n"
        "  --watch N        repeat every N seconds (clear screen), until Ctrl-C\n"
        "  --shutdown       send a Shutdown RPC to the daemon and exit\n",
        argv0, kDefaultSocket);
}

// Returns true on success, false on a usage error (message already printed).
bool parse_args(int argc, char** argv, Options* opt) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s requires an argument\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--socket") {
            const char* v = need_value("--socket");
            if (!v) return false;
            opt->socket_path = v;
        } else if (a == "--prometheus") {
            opt->prometheus = true;
        } else if (a == "--shutdown") {
            opt->shutdown = true;
        } else if (a == "--watch") {
            const char* v = need_value("--watch");
            if (!v) return false;
            char* endp = nullptr;
            long n = std::strtol(v, &endp, 10);
            if (endp == v || *endp != '\0' || n <= 0) {
                std::fprintf(stderr, "error: --watch expects a positive integer\n");
                return false;
            }
            opt->watch = true;
            opt->watch_seconds = static_cast<int>(n);
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n", a.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Minimal synchronous RPC client over uds + wire (mirrors RawClient in the
// daemon RPC test). One frame out, one frame back.
// ---------------------------------------------------------------------------
struct StatClient {
    int sock = -1;

    explicit StatClient(const std::string& path) {
        // uds_connect throws on failure/timeout; keep a short timeout so
        // --watch retries feel responsive.
        sock = uds_connect(path, 1000);
    }
    ~StatClient() { if (sock >= 0) close_fd(sock); }

    StatClient(const StatClient&) = delete;
    StatClient& operator=(const StatClient&) = delete;

    template <typename Resp>
    Resp call(OpCode op, const std::vector<uint8_t>& body,
              Resp (*dec)(const uint8_t*, size_t)) {
        send_frame(sock, make_frame(op, body), {});
        OpCode rop = OpCode::kInvalid;
        std::vector<uint8_t> payload;
        std::vector<int> fds;
        bool got = recv_frame(sock, &rop, &payload, &fds);
        for (int fd : fds) close_fd(fd);  // GetStats/Shutdown carry no fds
        if (!got) throw std::runtime_error("daemon closed connection");
        if (rop != op) {
            throw std::runtime_error("unexpected reply opcode " +
                                     std::to_string(static_cast<int>(rop)) +
                                     " (wanted " +
                                     std::to_string(static_cast<int>(op)) + ")");
        }
        return dec(payload.data(), payload.size());
    }

    GetStatsResponse get_stats() {
        GetStatsRequest req;  // rank_id 0 => default global snapshot
        return call<GetStatsResponse>(OpCode::kGetStats, encode(req),
                                      decode_GetStatsResponse);
    }

    ShutdownResponse shutdown() {
        ShutdownRequest req;  // token 0 accepted
        return call<ShutdownResponse>(OpCode::kShutdown, encode(req),
                                      decode_ShutdownResponse);
    }
};

// ---------------------------------------------------------------------------
// Classification helpers.
// ---------------------------------------------------------------------------
bool contains(const std::string& s, const char* sub) {
    return s.find(sub) != std::string::npos;
}
bool starts_with(const std::string& s, const char* pre) {
    size_t n = std::strlen(pre);
    return s.size() >= n && std::memcmp(s.data(), pre, n) == 0;
}

bool is_percentile(const std::string& key) {
    return contains(key, "bandwidth_mbps");
}
bool is_bytes(const std::string& key) {
    return contains(key, "bytes");
}
// Prometheus metric type: gauges are levels that go up and down; counters are
// monotonic. Byte occupancy/level metrics and the derived percentiles are
// gauges, everything else is a counter.
bool is_gauge(const std::string& key) {
    return is_bytes(key) || starts_with(key, "used_") ||
           starts_with(key, "inflight_") || starts_with(key, "draining_") ||
           is_percentile(key);
}

// Format a byte count with a human-readable binary-unit suffix.
std::string human_bytes(uint64_t v) {
    static const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"};
    double d = static_cast<double>(v);
    int u = 0;
    while (d >= 1024.0 && u < 6) { d /= 1024.0; ++u; }
    char buf[64];
    if (u == 0) {
        std::snprintf(buf, sizeof(buf), "%llu %s",
                      static_cast<unsigned long long>(v), units[u]);
    } else {
        std::snprintf(buf, sizeof(buf), "%.2f %s", d, units[u]);
    }
    return buf;
}

// ---------------------------------------------------------------------------
// Rendering.
// ---------------------------------------------------------------------------
void render_human(const GetStatsResponse& r) {
    if (!r.ok) {
        std::printf("daemon returned error: %s\n",
                    r.message.empty() ? "(no message)" : r.message.c_str());
        return;
    }
    if (!r.message.empty()) {
        std::printf("# %s\n", r.message.c_str());
    }

    // Widest key across all groups for a consistent left column.
    size_t keyw = 6;  // len("metric")
    for (const auto& k : r.keys) keyw = std::max(keyw, k.size());

    const size_t n = std::min(r.keys.size(), r.values.size());

    auto print_group = [&](const char* title, bool (*pred)(const std::string&),
                           bool show_human) {
        bool header_done = false;
        for (size_t i = 0; i < n; ++i) {
            const std::string& key = r.keys[i];
            if (!pred(key)) continue;
            if (!header_done) {
                std::printf("\n%s\n", title);
                std::printf("  %-*s  %20s\n", static_cast<int>(keyw), "metric",
                            "value");
                header_done = true;
            }
            uint64_t val = r.values[i];
            if (show_human) {
                std::printf("  %-*s  %20llu (%s)\n", static_cast<int>(keyw),
                            key.c_str(), static_cast<unsigned long long>(val),
                            human_bytes(val).c_str());
            } else {
                std::printf("  %-*s  %20llu\n", static_cast<int>(keyw),
                            key.c_str(), static_cast<unsigned long long>(val));
            }
        }
    };

    // Bytes group (with human-readable sizes), then plain counters, then the
    // derived bandwidth percentiles.
    print_group("bytes / occupancy",
                [](const std::string& k) { return is_bytes(k); }, true);
    print_group("counters",
                [](const std::string& k) {
                    return !is_bytes(k) && !is_percentile(k);
                },
                false);
    print_group("bandwidth (MB/s)",
                [](const std::string& k) { return is_percentile(k); }, false);
    std::printf("\n");
}

void render_prometheus(const GetStatsResponse& r) {
    if (!r.ok) {
        // Still emit a comment so scrapers see the tool ran.
        std::printf("# daemon returned error: %s\n",
                    r.message.empty() ? "(no message)" : r.message.c_str());
        return;
    }
    const size_t n = std::min(r.keys.size(), r.values.size());
    for (size_t i = 0; i < n; ++i) {
        const std::string& key = r.keys[i];  // already snake_case / valid
        std::string name = "offload_" + key;
        const char* type = is_gauge(key) ? "gauge" : "counter";
        std::printf("# HELP %s offload daemon metric %s\n", name.c_str(),
                    key.c_str());
        std::printf("# TYPE %s %s\n", name.c_str(), type);
        std::printf("%s %llu\n", name.c_str(),
                    static_cast<unsigned long long>(r.values[i]));
    }
}

std::string timestamp_now() {
    std::time_t t = std::time(nullptr);
    std::tm tmv{};
    localtime_r(&t, &tmv);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmv);
    return buf;
}

// One snapshot fetch + render. Returns true on success, false if the daemon
// was unreachable / errored (message already printed).
bool run_once(const Options& opt) {
    try {
        StatClient client(opt.socket_path);
        GetStatsResponse resp = client.get_stats();
        if (opt.prometheus) {
            render_prometheus(resp);
        } else {
            render_human(resp);
        }
        return resp.ok;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "daemon unreachable (%s): %s\n",
                     opt.socket_path.c_str(), e.what());
        return false;
    }
}

int do_shutdown(const Options& opt) {
    try {
        StatClient client(opt.socket_path);
        ShutdownResponse resp = client.shutdown();
        std::printf("shutdown: %s%s%s\n", resp.ok ? "ok" : "FAILED",
                    resp.message.empty() ? "" : " - ",
                    resp.message.c_str());
        return resp.ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "daemon unreachable (%s): %s\n",
                     opt.socket_path.c_str(), e.what());
        return 1;
    }
}

int do_watch(const Options& opt) {
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    while (!g_stop) {
        // Clear screen + home cursor.
        std::printf("\033[2J\033[H");
        std::printf("offload-stat --watch %ds   %s   socket=%s\n",
                    opt.watch_seconds, timestamp_now().c_str(),
                    opt.socket_path.c_str());
        std::fflush(stdout);
        // Reconnect every iteration: the daemon may restart between polls.
        run_once(opt);
        std::fflush(stdout);
        // Sleep in 1s slices so Ctrl-C is responsive.
        for (int s = 0; s < opt.watch_seconds && !g_stop; ++s) ::sleep(1);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) return 2;

    if (opt.shutdown) {
        if (opt.watch) {
            std::fprintf(stderr, "error: --shutdown and --watch are mutually "
                                 "exclusive\n");
            return 2;
        }
        return do_shutdown(opt);
    }

    if (opt.watch) return do_watch(opt);

    return run_once(opt) ? 0 : 1;
}
