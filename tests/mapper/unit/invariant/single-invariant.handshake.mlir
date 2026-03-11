module {
  handshake.func @single_invariant(%d: i1, %a: i32, ...) -> (i32)
      attributes {argNames = ["d", "a"], loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %inv = dataflow.invariant %d, %a : i1, i32 -> i32
    handshake.return %inv : i32
  }
}
