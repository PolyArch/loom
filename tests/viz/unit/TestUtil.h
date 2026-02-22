//===-- TestUtil.h - Viz unit test utilities -----------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_TEST_VIZ_TESTUTIL_H
#define LOOM_TEST_VIZ_TESTUTIL_H

#include <cstdio>
#include <cstdlib>
#include <string>

#define TEST_ASSERT(cond)                                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// Check that a string contains a substring.
#define TEST_CONTAINS(haystack, needle)                                         \
  do {                                                                         \
    if ((haystack).find(needle) == std::string::npos) {                        \
      fprintf(stderr, "FAIL: %s:%d: expected to find \"%s\" in output\n",      \
              __FILE__, __LINE__, needle);                                      \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// Check that a string does NOT contain a substring.
#define TEST_NOT_CONTAINS(haystack, needle)                                     \
  do {                                                                         \
    if ((haystack).find(needle) != std::string::npos) {                        \
      fprintf(stderr, "FAIL: %s:%d: expected NOT to find \"%s\" in output\n",  \
              __FILE__, __LINE__, needle);                                      \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#endif // LOOM_TEST_VIZ_TESTUTIL_H
