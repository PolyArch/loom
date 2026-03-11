module {
  handshake.func @single_cmpi_eq(%a: i32, %b: i32, ...) -> (i1)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = arith.cmpi eq, %a, %b : i32
    handshake.return %c : i1
  }
}
