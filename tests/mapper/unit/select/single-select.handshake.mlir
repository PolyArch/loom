module {
  handshake.func @single_select(%cond: i1, %a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["cond", "a", "b"], loom.annotations = ["loom.accel"], resNames = ["c"]} {
    %c = arith.select %cond, %a, %b : i32
    handshake.return %c : i32
  }
}
