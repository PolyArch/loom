//===-- TestUtil.h - Mapper unit test utilities --------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_TEST_MAPPER_TESTUTIL_H
#define LOOM_TEST_MAPPER_TESTUTIL_H

#include <cstdio>
#include <cstdlib>

#define TEST_ASSERT(cond)                                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#endif // LOOM_TEST_MAPPER_TESTUTIL_H
