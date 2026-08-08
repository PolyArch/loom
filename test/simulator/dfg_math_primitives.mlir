// RUN: loom-dfg-sim %s --graph math_abs_float --output %t.abs.json
// RUN: FileCheck %s --check-prefix=ABS < %t.abs.json
// RUN: loom-dfg-sim %s --graph math_sine --output %t.sin.json
// RUN: FileCheck %s --check-prefix=SIN < %t.sin.json
// RUN: loom-dfg-sim %s --graph math_rounding_float --output %t.round.json
// RUN: FileCheck %s --check-prefix=ROUND < %t.round.json
// RUN: loom-dfg-sim %s --graph math_roundeven_edges --output %t.roundeven.json
// RUN: FileCheck %s --check-prefix=EVEN < %t.roundeven.json
// RUN: loom-dfg-sim %s --graph math_integer_abs --output %t.int.json
// RUN: FileCheck %s --check-prefix=INT < %t.int.json
// RUN: loom-dfg-sim %s --graph math_integer_abs_poison --output %t.int-poison.json
// RUN: FileCheck %s --check-prefix=INT-POISON < %t.int-poison.json

// ABS-DAG: "workload": "math_abs_float"
// ABS-DAG: "status": "pass"
// ABS-DAG: "math.absf": 1
// ABS-DAG: "f32:3"

// SIN-DAG: "workload": "math_sine"
// SIN-DAG: "status": "pass"
// SIN-DAG: "math.sin": 1
// SIN-DAG: "f32:0"

// ROUND: "final_outputs": [
// ROUND-NEXT: "none",
// ROUND-NEXT: "f32:2",
// ROUND-NEXT: "f32:3",
// ROUND-NEXT: "f32:3",
// ROUND-NEXT: "f32:-2",
// ROUND-NEXT: "f32:2"
// ROUND-DAG: "workload": "math_rounding_float"
// ROUND-DAG: "status": "pass"
// ROUND-DAG: "math.floor": 1
// ROUND-DAG: "math.ceil": 1
// ROUND-DAG: "math.round": 1
// ROUND-DAG: "math.trunc": 1
// ROUND-DAG: "math.roundeven": 1

// EVEN: "final_outputs": [
// EVEN-NEXT: "none",
// EVEN-NEXT: "f32:-0",
// EVEN-NEXT: "f32:0",
// EVEN-NEXT: "f32:2",
// EVEN-NEXT: "f32:2",
// EVEN-NEXT: "f32:4"
// EVEN-DAG: "workload": "math_roundeven_edges"
// EVEN-DAG: "status": "pass"
// EVEN-DAG: "math.roundeven": 5

// INT-DAG: "workload": "math_integer_abs"
// INT-DAG: "status": "pass"
// INT-DAG: "math.absi": 1
// INT-DAG: "i32:17"

// INT-POISON-DAG: "workload": "math_integer_abs_poison"
// INT-POISON-DAG: "status": "pass"
// INT-POISON-DAG: "math.absi": 1
// INT-POISON-DAG: "i8:poison"

module {
  dataflow.graph private @math_abs_float(%ctrl: none) -> (f32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %neg = dataflow.constant %ctrl {const_value = -3.000000e+00 : f32} : f32
    %abs = math.absf %neg : f32
    %published:2 = dataflow.sync %ctrl, %abs
        : (none, f32) -> (none, f32)
    dataflow.graph.return %published#0, %published#1 : none, f32
  }

  dataflow.graph private @math_sine(%ctrl: none) -> (f32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %zero = dataflow.constant %ctrl {const_value = 0.000000e+00 : f32} : f32
    %sin = math.sin %zero
        {loom.special_math_accuracy = "CorrectlyRounded"} : f32
    %published:2 = dataflow.sync %ctrl, %sin
        : (none, f32) -> (none, f32)
    dataflow.graph.return %published#0, %published#1 : none, f32
  }

  dataflow.graph private @math_rounding_float(%ctrl: none)
      -> (f32, f32, f32, f32, f32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 5, 0, 0>} {
    %floor_in = dataflow.constant %ctrl {const_value = 2.750000e+00 : f32} : f32
    %floor = math.floor %floor_in
        {loom.special_math_accuracy = "CorrectlyRounded"} : f32
    %ceil_in = dataflow.constant %ctrl {const_value = 2.250000e+00 : f32} : f32
    %ceil = math.ceil %ceil_in
        {loom.special_math_accuracy = "CorrectlyRounded"} : f32
    %round_in = dataflow.constant %ctrl {const_value = 2.500000e+00 : f32} : f32
    %round = math.round %round_in
        {loom.special_math_accuracy = "CorrectlyRounded"} : f32
    %trunc_in = dataflow.constant %ctrl {const_value = -2.750000e+00 : f32} : f32
    %trunc = math.trunc %trunc_in
        {loom.special_math_accuracy = "CorrectlyRounded"} : f32
    %even = math.roundeven %round_in
        {loom.special_math_accuracy = "CorrectlyRounded"} : f32
    %published:6 = dataflow.sync %ctrl, %floor, %ceil, %round, %trunc, %even
        : (none, f32, f32, f32, f32, f32)
          -> (none, f32, f32, f32, f32, f32)
    dataflow.graph.return %published#0, %published#1, %published#2,
        %published#3, %published#4, %published#5
        : none, f32, f32, f32, f32, f32
  }

  dataflow.graph private @math_roundeven_edges(%ctrl: none)
      -> (f32, f32, f32, f32, f32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 5, 0, 0>} {
    %neg_half_in = dataflow.constant %ctrl {const_value = -5.000000e-01 : f32} : f32
    %neg_half = math.roundeven %neg_half_in
        {loom.special_math_accuracy = "CorrectlyRounded"} : f32
    %pos_half_in = dataflow.constant %ctrl {const_value = 5.000000e-01 : f32} : f32
    %pos_half = math.roundeven %pos_half_in
        {loom.special_math_accuracy = "CorrectlyRounded"} : f32
    %one_half_in = dataflow.constant %ctrl {const_value = 1.500000e+00 : f32} : f32
    %one_half = math.roundeven %one_half_in
        {loom.special_math_accuracy = "CorrectlyRounded"} : f32
    %two_half_in = dataflow.constant %ctrl {const_value = 2.500000e+00 : f32} : f32
    %two_half = math.roundeven %two_half_in
        {loom.special_math_accuracy = "CorrectlyRounded"} : f32
    %three_half_in = dataflow.constant %ctrl {const_value = 3.500000e+00 : f32} : f32
    %three_half = math.roundeven %three_half_in
        {loom.special_math_accuracy = "CorrectlyRounded"} : f32
    %published:6 = dataflow.sync %ctrl, %neg_half, %pos_half, %one_half,
        %two_half, %three_half
        : (none, f32, f32, f32, f32, f32)
          -> (none, f32, f32, f32, f32, f32)
    dataflow.graph.return %published#0, %published#1, %published#2,
        %published#3, %published#4, %published#5
        : none, f32, f32, f32, f32, f32
  }

  dataflow.graph private @math_integer_abs(%ctrl: none) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %neg = dataflow.constant %ctrl {const_value = -17 : i32} : i32
    %abs = math.absi %neg : i32
    %published:2 = dataflow.sync %ctrl, %abs
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }

  dataflow.graph private @math_integer_abs_poison(%ctrl: none)
      -> (i8)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %min = dataflow.constant %ctrl {const_value = -128 : i8} : i8
    %abs = math.absi %min : i8
    %published:2 = dataflow.sync %ctrl, %abs
        : (none, i8) -> (none, i8)
    dataflow.graph.return %published#0, %published#1 : none, i8
  }
}
