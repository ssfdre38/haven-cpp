#pragma once

#include <cstdlib>
#include <iostream>

// ============================================================================
// Release-Safe Test Assertion Macro (NEVER compiled out by -DNDEBUG!)
// ============================================================================
// Include this header AFTER <cassert> in test files so both HAVEN_CHECK(...)
// and legacy assert(...) calls remain enforced in -DCMAKE_BUILD_TYPE=Release.
// ============================================================================

#define HAVEN_CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "\n❌ ASSERTION FAILED: " #cond " at " __FILE__ ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while (0)

#ifdef assert
#undef assert
#endif
#define assert(cond) HAVEN_CHECK(cond)
