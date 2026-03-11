module {
  handshake.func @single_index_castui(%a: index, ...) -> (i32)
      attributes {argNames = ["a"], loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %0 = arith.index_castui %a : index to i32
    handshake.return %0 : i32
  }
}
