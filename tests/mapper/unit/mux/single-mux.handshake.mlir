module {
  handshake.func @single_mux(%sel: index, %a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["sel", "a", "b"], loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %r = handshake.mux %sel [%a, %b] : index, i32
    handshake.return %r : i32
  }
}
