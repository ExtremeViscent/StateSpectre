// Small shared utilities: monotonic/real time, logging, string helpers.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <string>

namespace offload {

// Nanoseconds from CLOCK_MONOTONIC (for durations) and CLOCK_REALTIME (for
// wall-clock trace timestamps / heartbeat comparison across processes).
inline uint64_t now_mono_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
}
inline uint64_t now_real_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
}

// Log levels controlled by env OFLD_LOG_LEVEL: 0=err 1=warn 2=info(default) 3=debug
enum class LogLevel { kError = 0, kWarn = 1, kInfo = 2, kDebug = 3 };

namespace detail {
inline LogLevel& log_level() {
    static LogLevel lvl = [] {
        const char* e = std::getenv("OFLD_LOG_LEVEL");
        int v = e ? std::atoi(e) : 2;
        if (v < 0) v = 0;
        if (v > 3) v = 3;
        return static_cast<LogLevel>(v);
    }();
    return lvl;
}
inline std::mutex& log_mutex() {
    static std::mutex m;
    return m;
}
inline const char* level_tag(LogLevel l) {
    switch (l) {
        case LogLevel::kError: return "ERROR";
        case LogLevel::kWarn:  return "WARN ";
        case LogLevel::kInfo:  return "INFO ";
        case LogLevel::kDebug: return "DEBUG";
    }
    return "?????";
}
}  // namespace detail

#define OFLD_LOG(level, tag, fmt, ...)                                            \
    do {                                                                          \
        if (static_cast<int>(level) <= static_cast<int>(::offload::detail::log_level())) { \
            std::lock_guard<std::mutex> _lg(::offload::detail::log_mutex());       \
            std::fprintf(stderr, "[%s][%s] " fmt "\n",                            \
                         ::offload::detail::level_tag(level), tag, ##__VA_ARGS__); \
        }                                                                         \
    } while (0)

#define OFLD_ERR(tag, fmt, ...)   OFLD_LOG(::offload::LogLevel::kError, tag, fmt, ##__VA_ARGS__)
#define OFLD_WARN(tag, fmt, ...)  OFLD_LOG(::offload::LogLevel::kWarn,  tag, fmt, ##__VA_ARGS__)
#define OFLD_INFO(tag, fmt, ...)  OFLD_LOG(::offload::LogLevel::kInfo,  tag, fmt, ##__VA_ARGS__)
#define OFLD_DEBUG(tag, fmt, ...) OFLD_LOG(::offload::LogLevel::kDebug, tag, fmt, ##__VA_ARGS__)

}  // namespace offload
