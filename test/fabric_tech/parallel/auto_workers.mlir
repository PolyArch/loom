// RUN: loom-parallel-test --workers 0 --workers-effective | FileCheck %s

// workers=0 must auto-detect via std::thread::hardware_concurrency(); we
// only assert the printed value is a positive integer (don't pin a number
// that varies across machines).

// CHECK: workers={{[1-9][0-9]*}}
