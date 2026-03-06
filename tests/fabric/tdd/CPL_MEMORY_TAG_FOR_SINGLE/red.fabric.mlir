// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MEMORY_TAG_FOR_SINGLE

// ldCount = 1 but load address port uses tagged type instead of native.
// Tagged load ports are not allowed when ldCount == 1.
fabric.module @tagged_single(
    %ldaddr: !dataflow.tagged<!dataflow.bits<57>, i1>
) -> (!dataflow.tagged<!dataflow.bits<32>, i1>, !dataflow.tagged<none, i1>) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0]
      (%ldaddr)
      : memref<64xi32>, (!dataflow.tagged<!dataflow.bits<57>, i1>) -> (!dataflow.tagged<!dataflow.bits<32>, i1>, !dataflow.tagged<none, i1>)
  fabric.yield %lddata, %lddone : !dataflow.tagged<!dataflow.bits<32>, i1>, !dataflow.tagged<none, i1>
}
