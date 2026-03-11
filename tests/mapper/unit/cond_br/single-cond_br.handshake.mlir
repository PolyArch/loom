module {
  handshake.func @single_cond_br(%cond: i1, %val: i32, ...) -> (i32, i32)
      attributes {argNames = ["cond", "val"], loom.annotations = ["loom.accel"],
                  resNames = ["true_out", "false_out"]} {
    %t, %f = handshake.cond_br %cond, %val : i32
    handshake.return %t, %f : i32, i32
  }
}
