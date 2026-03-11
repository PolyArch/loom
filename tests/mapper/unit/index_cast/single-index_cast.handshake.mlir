module {
  handshake.func @single_index_cast(%a: index, ...) -> (i32)
      attributes {argNames = ["a"], loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %0 = arith.index_cast %a : index to i32
    handshake.return %0 : i32
  }
}
