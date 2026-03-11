module {
  handshake.func @single_sink(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    handshake.sink %a : i32
    handshake.return %b : i32
  }
}
