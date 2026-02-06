// RUN: not loom --adg %s 2>&1 | FileCheck %s
// XFAIL: *
// CHECK: COMP_MEMORY_TAG_WIDTH

// ldCount = 4 requires tag width >= 2 (log2Ceil(4) = 2), but i1 is provided.
// Output layout: [lddata * 4] [lddone] = 5 outputs.
fabric.module @narrow_tag(
    %a0: !dataflow.tagged<index, i1>,
    %a1: !dataflow.tagged<index, i1>,
    %a2: !dataflow.tagged<index, i1>,
    %a3: !dataflow.tagged<index, i1>
) -> (!dataflow.tagged<i32, i1>, !dataflow.tagged<i32, i1>,
      !dataflow.tagged<i32, i1>, !dataflow.tagged<i32, i1>,
      !dataflow.tagged<none, i1>) {
  %d0, %d1, %d2, %d3, %done = fabric.memory
      [ldCount = 4, stCount = 0]
      (%a0, %a1, %a2, %a3)
      : memref<64xi32>,
        (!dataflow.tagged<index, i1>, !dataflow.tagged<index, i1>,
         !dataflow.tagged<index, i1>, !dataflow.tagged<index, i1>)
        -> (!dataflow.tagged<i32, i1>, !dataflow.tagged<i32, i1>,
            !dataflow.tagged<i32, i1>, !dataflow.tagged<i32, i1>,
            !dataflow.tagged<none, i1>)
  fabric.yield %d0, %d1, %d2, %d3, %done
      : !dataflow.tagged<i32, i1>, !dataflow.tagged<i32, i1>,
        !dataflow.tagged<i32, i1>, !dataflow.tagged<i32, i1>,
        !dataflow.tagged<none, i1>
}
