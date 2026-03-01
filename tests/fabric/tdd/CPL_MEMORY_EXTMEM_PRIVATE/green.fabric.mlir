// RUN: loom --adg %s

// A valid fabric.extmemory without is_private attribute, inside a fabric.module.
fabric.module @valid_ext(%m: memref<?xi32>, %addr: !dataflow.bits<57>) -> (!dataflow.bits<32>, none) {
  %lddata, %lddone = fabric.extmemory
      [ldCount = 1, stCount = 0]
      (%m, %addr)
      : memref<?xi32>, (memref<?xi32>, !dataflow.bits<57>) -> (!dataflow.bits<32>, none)
  fabric.yield %lddata, %lddone : !dataflow.bits<32>, none
}
