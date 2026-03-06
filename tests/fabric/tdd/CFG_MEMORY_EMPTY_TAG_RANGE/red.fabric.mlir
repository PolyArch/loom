// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CFG_MEMORY_EMPTY_TAG_RANGE

// Invalid: end_tag (2) <= start_tag (5) in entry 0 -> empty half-open range.
fabric.module @empty_range(%ldaddr: !dataflow.bits<57>) -> (!dataflow.bits<32>, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0, numRegion = 1,
       addr_offset_table = array<i64: 1, 5, 2, 0>]
      (%ldaddr)
      : memref<64xi32>, (!dataflow.bits<57>) -> (!dataflow.bits<32>, none)
  fabric.yield %lddata, %lddone : !dataflow.bits<32>, none
}
