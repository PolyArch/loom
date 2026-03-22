// Companion DFG for add_latency1.fabric.mlir behaviour test.
// Simple addition kernel that maps to the addi function unit.
module {
  handshake.func @add_test(%a: i32, %b: i32) -> (i32)
      attributes {argNames = ["a", "b"], resNames = ["result"]} {
    %0 = arith.addi %a, %b : i32
    handshake.return %0 : i32
  }
}
