module {
  handshake.func @single_carry(%ac: i1, %init: i32, %next: i32, ...) -> (i32)
      attributes {argNames = ["ac", "init", "next"], loom.annotations = ["loom.accel"],
                  resNames = ["val"]} {
    %val = dataflow.carry %ac, %init, %next : i1, i32, i32 -> i32
    handshake.return %val : i32
  }
}
