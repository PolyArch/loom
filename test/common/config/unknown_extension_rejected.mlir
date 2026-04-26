// RUN: not loom-config-test %p/unknown_extension_rejected.txt 2>&1 | FileCheck %s

// CHECK: error: unrecognized config extension '.txt'
