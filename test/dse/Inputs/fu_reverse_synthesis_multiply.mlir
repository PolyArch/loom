module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @multiply(%start: none, %lhs: i32, %rhs: i32) -> i32
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = arith.muli %lhs, %rhs : i32
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %lhs: i32, %rhs: i32) ctrl (%ctrl: none) {
    %value, %done = dataflow.graph.launch @multiply deps(%ctrl)
        values(%lhs, %rhs) stream_inputs() memories() stream_outputs()
        : (none, i32, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func @application() {
    %lhs = arith.constant 6 : i32
    %rhs = arith.constant 7 : i32
    %thread = dataflow.thread.launch @worker(%lhs, %rhs)
        : (i32, i32) -> !dataflow.thread_token
    return
  }
}
