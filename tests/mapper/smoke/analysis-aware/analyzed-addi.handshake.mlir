module {
  handshake.func @analyzed_addi(%a: i32, %b: i32, ...) -> (i32)
      attributes {argNames = ["a", "b"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %0 = arith.addi %a, %b {
      "loom.analysis" = {
        loop_depth = 1 : i32,
        exec_freq = 256 : i64,
        on_recurrence = true,
        recurrence_id = 0 : i32,
        on_critical_path = true,
        temporal_score = 0.05 : f64
      }
    } : i32
    handshake.return %0 : i32
  }
}
