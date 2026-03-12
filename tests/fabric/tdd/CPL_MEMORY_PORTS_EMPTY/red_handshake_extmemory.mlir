// RUN: mkdir -p %S/Output
// RUN: rm -f %S/Output/red_handshake_extmemory.fabric.mlir
// RUN: not loom --gen-adg --dfgs %s -o %S/Output/red_handshake_extmemory.fabric.mlir 2>&1 | FileCheck %s
// CHECK: CPL_MEMORY_PORTS_EMPTY

module {
  handshake.func @bad_extmem(%mem: memref<?xi32>, %start: none) -> none {
    handshake.extmemory[ld = 0, st = 0] (%mem : memref<?xi32>) () {id = 0 : i32} : () -> ()
    handshake.return %start : none
  }
}
