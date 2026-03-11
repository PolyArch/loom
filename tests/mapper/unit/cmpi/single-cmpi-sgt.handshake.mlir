module {
  handshake.func @single_cmpi_sgt(%a: i32, %b: i32, ...) -> (i1)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = arith.cmpi sgt, %a, %b : i32
    handshake.return %c : i1
  }
}
