// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_MEMORY_TAG_FOR_SINGLE

// ldCount = 1 but load address port uses tagged type instead of native.
// Tagged load ports are not allowed when ldCount == 1.
fabric.module @tagged_single(
    %ldaddr: !dataflow.tagged<index, i1>
) -> (!dataflow.tagged<i32, i1>, !dataflow.tagged<none, i1>) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0]
      (%ldaddr)
      : memref<64xi32>, (!dataflow.tagged<index, i1>) -> (!dataflow.tagged<i32, i1>, !dataflow.tagged<none, i1>)
  fabric.yield %lddata, %lddone : !dataflow.tagged<i32, i1>, !dataflow.tagged<none, i1>
}
