// Two graph pairs that share one multiply-accumulate-with-zero-point shape and
// differ only in the tail operation that consumes it.
//
// The int8 pair carries the widening form the quantized dot product lowers to,
// so mining must report the six-actor common subgraph. The i64 pair carries the
// same shape without the widening casts, so its four-actor common subgraph is
// inside the implementation families whose canonical capability derivation has
// an inverse policy and can be synthesized back to Fabric.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @dot_int8_shift(%start: none, %a: i8, %zp_a: i64,
                                         %b: i8, %zp_b: i64, %acc: i64,
                                         %shift: i64) -> i64
      attributes {input_segments = array<i32: 6, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %wide_a = arith.extsi %a : i8 to i64
    %wide_b = arith.extsi %b : i8 to i64
    %centered_a = arith.subi %wide_a, %zp_a : i64
    %centered_b = arith.subi %wide_b, %zp_b : i64
    %product = arith.muli %centered_a, %centered_b : i64
    %sum = arith.addi %product, %acc : i64
    %scaled = arith.shrsi %sum, %shift : i64
    %result:2 = dataflow.sync %start, %scaled : (none, i64) -> (none, i64)
    dataflow.graph.return values(%result#1 : i64) streams() memories()
        complete(%result#0 : none)
  }

  dataflow.graph private @dot_int8_clamp(%start: none, %a: i8, %zp_a: i64,
                                         %b: i8, %zp_b: i64, %acc: i64,
                                         %floor: i64) -> i64
      attributes {input_segments = array<i32: 6, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %wide_a = arith.extsi %a : i8 to i64
    %wide_b = arith.extsi %b : i8 to i64
    %centered_a = arith.subi %wide_a, %zp_a : i64
    %centered_b = arith.subi %wide_b, %zp_b : i64
    %product = arith.muli %centered_a, %centered_b : i64
    %sum = arith.addi %product, %acc : i64
    %clamped = arith.maxsi %sum, %floor : i64
    %result:2 = dataflow.sync %start, %clamped : (none, i64) -> (none, i64)
    dataflow.graph.return values(%result#1 : i64) streams() memories()
        complete(%result#0 : none)
  }

  dataflow.graph private @dot_shift(%start: none, %a: i64, %zp_a: i64,
                                    %b: i64, %zp_b: i64, %acc: i64,
                                    %shift: i64) -> i64
      attributes {input_segments = array<i32: 6, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %centered_a = arith.subi %a, %zp_a : i64
    %centered_b = arith.subi %b, %zp_b : i64
    %product = arith.muli %centered_a, %centered_b : i64
    %sum = arith.addi %product, %acc : i64
    %scaled = arith.shrsi %sum, %shift : i64
    %result:2 = dataflow.sync %start, %scaled : (none, i64) -> (none, i64)
    dataflow.graph.return values(%result#1 : i64) streams() memories()
        complete(%result#0 : none)
  }

  dataflow.graph private @dot_clamp(%start: none, %a: i64, %zp_a: i64,
                                    %b: i64, %zp_b: i64, %acc: i64,
                                    %floor: i64) -> i64
      attributes {input_segments = array<i32: 6, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %centered_a = arith.subi %a, %zp_a : i64
    %centered_b = arith.subi %b, %zp_b : i64
    %product = arith.muli %centered_a, %centered_b : i64
    %sum = arith.addi %product, %acc : i64
    %clamped = arith.maxsi %sum, %floor : i64
    %result:2 = dataflow.sync %start, %clamped : (none, i64) -> (none, i64)
    dataflow.graph.return values(%result#1 : i64) streams() memories()
        complete(%result#0 : none)
  }

  dataflow.thread private @dot_int8_shift_worker domain(#dataflow.thread_domain<dense>)(
      %a: i8, %zp_a: i64, %b: i8, %zp_b: i64, %acc: i64, %shift: i64)
      ctrl (%ctrl: none) {
    %value, %done = dataflow.graph.launch @dot_int8_shift deps(%ctrl)
        values(%a, %zp_a, %b, %zp_b, %acc, %shift) stream_inputs() memories()
        stream_outputs() : (none, i8, i64, i8, i64, i64, i64) -> (i64, none)
    dataflow.thread.yield %done : none
  }

  dataflow.thread private @dot_int8_clamp_worker domain(#dataflow.thread_domain<dense>)(
      %a: i8, %zp_a: i64, %b: i8, %zp_b: i64, %acc: i64, %floor: i64)
      ctrl (%ctrl: none) {
    %value, %done = dataflow.graph.launch @dot_int8_clamp deps(%ctrl)
        values(%a, %zp_a, %b, %zp_b, %acc, %floor) stream_inputs() memories()
        stream_outputs() : (none, i8, i64, i8, i64, i64, i64) -> (i64, none)
    dataflow.thread.yield %done : none
  }

  dataflow.thread private @dot_shift_worker domain(#dataflow.thread_domain<dense>)(
      %a: i64, %zp_a: i64, %b: i64, %zp_b: i64, %acc: i64, %shift: i64)
      ctrl (%ctrl: none) {
    %value, %done = dataflow.graph.launch @dot_shift deps(%ctrl)
        values(%a, %zp_a, %b, %zp_b, %acc, %shift) stream_inputs() memories()
        stream_outputs() : (none, i64, i64, i64, i64, i64, i64) -> (i64, none)
    dataflow.thread.yield %done : none
  }

  dataflow.thread private @dot_clamp_worker domain(#dataflow.thread_domain<dense>)(
      %a: i64, %zp_a: i64, %b: i64, %zp_b: i64, %acc: i64, %floor: i64)
      ctrl (%ctrl: none) {
    %value, %done = dataflow.graph.launch @dot_clamp deps(%ctrl)
        values(%a, %zp_a, %b, %zp_b, %acc, %floor) stream_inputs() memories()
        stream_outputs() : (none, i64, i64, i64, i64, i64, i64) -> (i64, none)
    dataflow.thread.yield %done : none
  }

  // Every graph is launched once so canonical finalization keeps it.
  func.func @application() {
    %a8 = arith.constant 3 : i8
    %b8 = arith.constant 5 : i8
    %a = arith.constant 3 : i64
    %b = arith.constant 5 : i64
    %zp = arith.constant 1 : i64
    %acc = arith.constant 7 : i64
    %tail = arith.constant 2 : i64
    %dot_int8_shift_thread = dataflow.thread.launch @dot_int8_shift_worker(%a8, %zp, %b8, %zp, %acc, %tail)
        : (i8, i64, i8, i64, i64, i64) -> !dataflow.thread_token
    %dot_int8_clamp_thread = dataflow.thread.launch @dot_int8_clamp_worker(%a8, %zp, %b8, %zp, %acc, %tail)
        : (i8, i64, i8, i64, i64, i64) -> !dataflow.thread_token
    %dot_shift_thread = dataflow.thread.launch @dot_shift_worker(%a, %zp, %b, %zp, %acc, %tail)
        : (i64, i64, i64, i64, i64, i64) -> !dataflow.thread_token
    %dot_clamp_thread = dataflow.thread.launch @dot_clamp_worker(%a, %zp, %b, %zp, %acc, %tail)
        : (i64, i64, i64, i64, i64, i64) -> !dataflow.thread_token
    return
  }
}
