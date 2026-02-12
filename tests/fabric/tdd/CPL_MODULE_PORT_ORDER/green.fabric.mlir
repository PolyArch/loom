// RUN: loom --adg %s

// A valid fabric.module with correct port ordering: native then tagged.
fabric.module @order_ok(%a: i32, %b: !dataflow.tagged<i32, i4>) -> (i32) {
  // Use switch broadcast to duplicate %a for two PE inputs
  %bcast:2 = fabric.switch [connectivity_table = [1, 1]] %a : i32 -> i32, i32
  %sum = fabric.pe %bcast#0, %bcast#1 : (i32, i32) -> (i32) {
  ^bb0(%x: i32, %y: i32):
    %r = arith.addi %x, %y : i32
    fabric.yield %r : i32
  }
  fabric.yield %sum : i32
}
