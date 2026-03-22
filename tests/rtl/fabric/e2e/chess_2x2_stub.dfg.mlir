// Companion DFG for chess_2x2_stub.fabric.mlir end-to-end RTL test.
//
// This DFG computes result = (a + b) * c, which maps to the 2x2 chess
// grid topology: pe_add computes a+b, pe_mul computes (a+b)*c.
//
// Port alignment with ADG:
//   DFG inputs:  a(i32), b(i32), c(i32) -> ADG inputs: in0, in1, in2
//   DFG output:  result(i32)            -> ADG output: out0
//
// Adapted from tests/unit/chess-2x2/dfg.mlir.
module {
  handshake.func @chess_2x2_test(%a: i32, %b: i32, %c: i32) -> (i32)
      attributes {argNames = ["a", "b", "c"], resNames = ["result"]} {
    %0 = arith.addi %a, %b : i32
    %1 = arith.muli %0, %c : i32
    handshake.return %1 : i32
  }
}
