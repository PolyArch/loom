// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_MEMORY_TAG_WIDTH

// ldCount = 4 requires tag width >= 2 (log2Ceil(4) = 2), but i1 is provided.
// Singular port layout: 1 input (ld_addr), 2 outputs (ld_data, ld_done).
fabric.module @narrow_tag(
    %ld_addr: !dataflow.tagged<index, i1>
) -> (!dataflow.tagged<i32, i1>, !dataflow.tagged<none, i1>) {
  %ld_data, %ld_done = fabric.memory
      [ldCount = 4, stCount = 0]
      (%ld_addr)
      : memref<64xi32>,
        (!dataflow.tagged<index, i1>)
        -> (!dataflow.tagged<i32, i1>, !dataflow.tagged<none, i1>)
  fabric.yield %ld_data, %ld_done
      : !dataflow.tagged<i32, i1>, !dataflow.tagged<none, i1>
}
