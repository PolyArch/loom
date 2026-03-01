// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MEMORY_DATA_NATIVE

// Named memory with native i32 store data port (must use bits<32>).
// Top-level named form avoids module-boundary interception.
fabric.memory @bad_native_store_data
    [ldCount = 0, stCount = 1, lsqDepth = 4]
    : memref<64xi32>,
      (!dataflow.bits<57>, i32) -> (none)
