// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MODULE_NATIVE_PORT

// Memory data port uses native i32 (must use bits<32>).
// The module-boundary check (CPL_MODULE_NATIVE_PORT) fires before
// the memory-level check (CPL_MEMORY_DATA_NATIVE) because native
// types in module outputs are caught first.
fabric.module @bad_mem_data(
    %ldaddr: !dataflow.bits<57>
) -> (i32, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0]
      (%ldaddr)
      : memref<64xi32>, (!dataflow.bits<57>) -> (i32, none)
  fabric.yield %lddata, %lddone : i32, none
}
