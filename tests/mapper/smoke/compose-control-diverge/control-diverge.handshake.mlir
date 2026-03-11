module {
  handshake.func @control_diverge(%cond: i1, %a: i32, %b: i32, ...) -> (i32, i32)
      attributes {argNames = ["cond", "a", "b"], loom.annotations = ["loom.accel"],
                  resNames = ["true_out", "false_out"]} {
    %t, %f = handshake.cond_br %cond, %a : i32
    %t2 = arith.addi %t, %b : i32
    %f2 = arith.subi %f, %b : i32
    handshake.return %t2, %f2 : i32, i32
  }
}
