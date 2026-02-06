// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_PE_LOADSTORE_BODY

// PE body has handshake.load PLUS an extra arith.addi, violating
// the exclusivity rule: load/store PE body must contain exactly one
// handshake.load or handshake.store and no other non-terminator ops.
fabric.module @test(%addr: index, %data: i32, %ctrl: none) -> (i32, index) {
  %d, %a = fabric.pe %addr, %data, %ctrl : (index, i32, none) -> (i32, index) {
  ^bb0(%x: index, %y: i32, %c: none):
    %ld_d, %ld_a = handshake.load [%x] %y, %c : index, i32
    %extra = arith.addi %ld_d, %ld_d : i32
    fabric.yield %extra, %ld_a : i32, index
  }
  fabric.yield %d, %a : i32, index
}
