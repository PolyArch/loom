// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MEMORY_EXTMEM_BINDING

// A fabric.module with inline fabric.extmemory where the memref operand
// comes from a fabric.memory result, not from a module block argument.
fabric.module @bad_extmem_binding(%addr: !dataflow.bits<57>) -> (!dataflow.bits<32>, none) {
  // Broadcast %addr for two consumers (memory and extmemory)
  %bcast_addr:2 = fabric.switch [connectivity_table = [1, 1]] %addr : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>
  %mem, %ld1, %done1 = fabric.memory
      [ldCount = 1, stCount = 0, is_private = false]
      (%bcast_addr#0)
      : memref<64xi32>, (!dataflow.bits<57>) -> (memref<64xi32>, !dataflow.bits<32>, none)
  %lddata, %lddone = fabric.extmemory
      [ldCount = 1, stCount = 0]
      (%mem, %bcast_addr#1)
      : memref<64xi32>, (memref<64xi32>, !dataflow.bits<57>) -> (!dataflow.bits<32>, none)
  fabric.yield %lddata, %lddone : !dataflow.bits<32>, none
}
