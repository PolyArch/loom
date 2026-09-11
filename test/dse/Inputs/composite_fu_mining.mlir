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
}
