// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CFG_MEMORY_OVERLAP_TAG_REGION

// This test validates a configuration that would cause RT_MEMORY_NO_MATCH at
// runtime for tags in the gap [3,5). At compile time, the configuration is
// invalid for a different reason: the overlapping regions [0,5) and [3,8)
// overlap at [3,5), triggering CFG_MEMORY_OVERLAP_TAG_REGION.
// A properly gapped configuration (e.g., [0,3) and [5,8)) is valid MLIR
// but would produce RT_MEMORY_NO_MATCH for tags 3,4 at runtime.
fabric.module @gap_with_overlap(%ldaddr: !dataflow.bits<57>) -> (!dataflow.bits<32>, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0, numRegion = 2,
       addr_offset_table = array<i64: 1, 0, 5, 0,  1, 3, 8, 64>]
      (%ldaddr)
      : memref<128xi32>, (!dataflow.bits<57>) -> (!dataflow.bits<32>, none)
  fabric.yield %lddata, %lddone : !dataflow.bits<32>, none
}
