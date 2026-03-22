// Companion DFG for single_fu_add.fabric.mlir behaviour test.
// Simple addition kernel mapping to the spatial PE with one addi FU.
module {
  handshake.func @add_pe_test(%a: i32, %b: i32) -> (i32)
      attributes {argNames = ["a", "b"], resNames = ["result"]} {
    %0 = arith.addi %a, %b : i32
    handshake.return %0 : i32
  }
}
