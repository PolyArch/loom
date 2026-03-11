module {
  handshake.func @cmp_select(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %cmp = arith.cmpi slt, %a, %b : i32
    %sel = arith.select %cmp, %a, %b : i32
    handshake.return %sel : i32
  }
}
