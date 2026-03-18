module {
  handshake.func @dataflow_invariant_test(%cond: i1, %value: i32, ...) -> (i32)
      attributes {argNames = ["cond", "value"], resNames = ["out"]} {
    %0 = dataflow.invariant %cond, %value : i1, i32 -> i32
    return %0 : i32
  }
}
