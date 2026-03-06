// RUN: loom --adg %s

// A fabric.module that yields a memref produced by a non-private fabric.memory.
fabric.module @export_mem(%in_addr: !dataflow.bits<57>) -> (memref<64xi32>, !dataflow.bits<32>, none) {
  %mem, %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0, is_private = false]
      (%in_addr)
      : memref<64xi32>, (!dataflow.bits<57>) -> (memref<64xi32>, !dataflow.bits<32>, none)
  fabric.yield %mem, %lddata, %lddone : memref<64xi32>, !dataflow.bits<32>, none
}
