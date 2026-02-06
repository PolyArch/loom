// RUN: loom --adg %s

// A valid fabric.module with correct port ordering: native then tagged.
fabric.module @order_ok(%a: i32, %b: !dataflow.tagged<i32, i4>) -> (i32) {
  %sum = fabric.pe %a, %a : (i32, i32) -> (i32) {
  ^bb0(%x: i32, %y: i32):
    %r = arith.addi %x, %y : i32
    fabric.yield %r : i32
  }
  fabric.yield %sum : i32
}
