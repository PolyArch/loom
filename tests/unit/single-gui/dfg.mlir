// Minimal DFG: two ops (addi + muli)
module {
  handshake.func @tiny(%a: i32, %b: i32, %ctrl: none, ...) -> (i32, none) attributes {argNames = ["a", "b", "ctrl"], resNames = ["result", "done"]} {
    %sum = arith.addi %a, %b : i32
    %prod = arith.muli %sum, %a : i32
    return %prod, %ctrl : i32, none
  }
}
