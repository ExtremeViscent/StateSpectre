// Minimal test harness: no external dependency. CHECK/CHECK_EQ print and count
// failures; main returns nonzero if any failed. Keeps tests self-contained.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace ofldtest {

inline int& fail_count() { static int n = 0; return n; }
inline int& check_count() { static int n = 0; return n; }

#define CHECK(cond)                                                            \
    do {                                                                       \
        ::ofldtest::check_count()++;                                           \
        if (!(cond)) {                                                         \
            ::ofldtest::fail_count()++;                                        \
            std::fprintf(stderr, "  FAIL %s:%d: CHECK(%s)\n", __FILE__,        \
                         __LINE__, #cond);                                     \
        }                                                                      \
    } while (0)

#define CHECK_EQ(a, b)                                                         \
    do {                                                                       \
        ::ofldtest::check_count()++;                                           \
        auto _va = (a);                                                        \
        auto _vb = (b);                                                        \
        if (!(_va == _vb)) {                                                   \
            ::ofldtest::fail_count()++;                                        \
            std::fprintf(stderr, "  FAIL %s:%d: CHECK_EQ(%s == %s) => %lld vs %lld\n", \
                         __FILE__, __LINE__, #a, #b,                           \
                         (long long)_va, (long long)_vb);                      \
        }                                                                      \
    } while (0)

#define CHECK_STREQ(a, b)                                                      \
    do {                                                                       \
        ::ofldtest::check_count()++;                                           \
        if (std::string(a) != std::string(b)) {                               \
            ::ofldtest::fail_count()++;                                        \
            std::fprintf(stderr, "  FAIL %s:%d: CHECK_STREQ(\"%s\" == \"%s\")\n",\
                         __FILE__, __LINE__, std::string(a).c_str(),           \
                         std::string(b).c_str());                             \
        }                                                                      \
    } while (0)

#define RUN(fn)                                                                \
    do {                                                                       \
        std::fprintf(stderr, "[run] %s\n", #fn);                              \
        fn();                                                                  \
    } while (0)

inline int summary(const char* suite) {
    if (fail_count() == 0) {
        std::fprintf(stderr, "[PASS] %s: %d checks OK\n", suite, check_count());
        return 0;
    }
    std::fprintf(stderr, "[FAIL] %s: %d/%d checks failed\n", suite,
                 fail_count(), check_count());
    return 1;
}

}  // namespace ofldtest
