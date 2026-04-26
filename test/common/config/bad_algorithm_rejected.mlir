// RUN: not loom-config-test %p/bad_algorithm_rejected.yaml 2>&1 | FileCheck %s

// CHECK: error: techmap.algorithm must be one of greedy|list|beam|sa, got 'ilp'
