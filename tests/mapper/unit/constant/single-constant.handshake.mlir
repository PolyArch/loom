module {
  handshake.func @single_constant(%ctrl: none, ...) -> (i32)
      attributes {argNames = ["ctrl"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = handshake.constant %ctrl {value = 42 : i32} : i32
    handshake.return %c : i32
  }
}
