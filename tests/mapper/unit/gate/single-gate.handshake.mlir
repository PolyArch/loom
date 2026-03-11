module {
  handshake.func @single_gate(%val: i32, %cond: i1, ...) -> (i32, i1)
      attributes {argNames = ["val", "cond"], loom.annotations = ["loom.accel"],
                  resNames = ["after_val", "after_cond"]} {
    %afterVal, %afterCond = dataflow.gate %val, %cond : i32, i1 -> i32, i1
    handshake.return %afterVal, %afterCond : i32, i1
  }
}
