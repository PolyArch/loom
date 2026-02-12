// RUN: loom --adg %s

// A valid fabric.memory with ldCount = 4 and tag width i2 (log2Ceil(4) = 2).
// Output layout: [lddata * 4] [lddone]
fabric.module @valid_tag_width(
    %a0: !dataflow.tagged<index, i2>,
    %a1: !dataflow.tagged<index, i2>,
    %a2: !dataflow.tagged<index, i2>,
    %a3: !dataflow.tagged<index, i2>
) -> (!dataflow.tagged<i32, i2>, !dataflow.tagged<i32, i2>,
      !dataflow.tagged<i32, i2>, !dataflow.tagged<i32, i2>,
      !dataflow.tagged<none, i2>) {
  %d0, %d1, %d2, %d3, %done = fabric.memory
      [ldCount = 4, stCount = 0]
      (%a0, %a1, %a2, %a3)
      : memref<64xi32>,
        (!dataflow.tagged<index, i2>, !dataflow.tagged<index, i2>,
         !dataflow.tagged<index, i2>, !dataflow.tagged<index, i2>)
        -> (!dataflow.tagged<i32, i2>, !dataflow.tagged<i32, i2>,
            !dataflow.tagged<i32, i2>, !dataflow.tagged<i32, i2>,
            !dataflow.tagged<none, i2>)
  fabric.yield %d0, %d1, %d2, %d3, %done
      : !dataflow.tagged<i32, i2>, !dataflow.tagged<i32, i2>,
        !dataflow.tagged<i32, i2>, !dataflow.tagged<i32, i2>,
        !dataflow.tagged<none, i2>
}
