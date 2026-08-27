module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @select(%start: none, %lhs: i32, %rhs: i32) -> i32
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %condition = arith.cmpi slt, %lhs, %rhs : i32
    %value = arith.select %condition, %lhs, %rhs : i32
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %lhs: i32, %rhs: i32) ctrl (%ctrl: none) {
    %value, %done = dataflow.graph.launch @select deps(%ctrl)
        values(%lhs, %rhs) stream_inputs() memories() stream_outputs()
        : (none, i32, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func @application() {
    %lhs = arith.constant 19 : i32
    %rhs = arith.constant 7 : i32
    %thread = dataflow.thread.launch @worker(%lhs, %rhs)
        : (i32, i32) -> !dataflow.thread_token
    return
  }
}
