// Simple DFG for GUI visualization testing.
// Computes: result = (a + b) * 2
module {
  handshake.func @all_pes_test(%arg0: i32, %arg1: i32, %arg2: none, ...) -> (i32, none) attributes {argNames = ["a", "b", "ctrl"], resNames = ["result", "done"]} {
    %sum = arith.addi %arg0, %arg1 : i32
    %c2 = handshake.constant %arg2 {value = 2 : i32} : i32
    %product = arith.muli %sum, %c2 : i32
    %done = handshake.join %arg2 : none
    return %product, %done : i32, none
  }
}
