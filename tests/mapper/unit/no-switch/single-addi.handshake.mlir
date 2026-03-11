module {
  handshake.func @single_addi(%a: i32, %b: i32, %ctrl: none, ...) -> (i32, none)
      attributes {argNames = ["a", "b", "ctrl"], loom.annotations = ["loom.accel"],
                  resNames = ["sum", "done"]} {
    %sum = arith.addi %a, %b : i32
    handshake.return %sum, %ctrl : i32, none
  }
}
