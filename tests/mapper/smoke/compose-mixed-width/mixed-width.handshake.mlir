module {
  handshake.func @mixed_width_extsi_add(%a: i32, %b: i32, ...) -> (i64)
      attributes {argNames = ["a", "b"], loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %ea = arith.extsi %a : i32 to i64
    %eb = arith.extsi %b : i32 to i64
    %out = arith.addi %ea, %eb : i64
    handshake.return %out : i64
  }
}
