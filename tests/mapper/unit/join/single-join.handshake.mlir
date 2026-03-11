module {
  handshake.func @single_join(%a: i32, %b: i32, ...) -> (none)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %0 = handshake.join %a, %b : i32, i32
    handshake.return %0 : none
  }
}
