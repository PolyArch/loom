// RUN: loom-dfg-sim %s --graph math_unary_float --output %t.float.json
// RUN: FileCheck %s --check-prefix=FLOAT < %t.float.json
// RUN: loom-dfg-sim %s --graph math_rounding_float --output %t.round.json
// RUN: FileCheck %s --check-prefix=ROUND < %t.round.json
// RUN: loom-dfg-sim %s --graph math_roundeven_edges --output %t.roundeven.json
// RUN: FileCheck %s --check-prefix=EVEN < %t.roundeven.json
// RUN: loom-dfg-sim %s --graph math_integer_abs --output %t.int.json
// RUN: FileCheck %s --check-prefix=INT < %t.int.json
// RUN: loom-dfg-sim %s --graph math_integer_abs_poison --output %t.int-poison.json
// RUN: FileCheck %s --check-prefix=INT-POISON < %t.int-poison.json

// FLOAT-DAG: "workload": "math_unary_float"
// FLOAT-DAG: "status": "pass"
// FLOAT-DAG: "math.absf": 1
// FLOAT-DAG: "math.sqrt": 1
// FLOAT-DAG: "math.rsqrt": 1
// FLOAT-DAG: "math.sin": 1
// FLOAT-DAG: "math.cos": 1
// FLOAT-DAG: "math.tan": 1
// FLOAT-DAG: "math.sinh": 1
// FLOAT-DAG: "math.cosh": 1
// FLOAT-DAG: "math.tanh": 1
// FLOAT-DAG: "math.exp": 1
// FLOAT-DAG: "math.exp2": 1
// FLOAT-DAG: "math.expm1": 1
// FLOAT-DAG: "math.log": 1
// FLOAT-DAG: "math.log2": 1
// FLOAT-DAG: "math.log10": 1
// FLOAT-DAG: "math.log1p": 1
// FLOAT-DAG: "math.erf": 1
// FLOAT-DAG: "f32:3"
// FLOAT-DAG: "f32:4"
// FLOAT-DAG: "f32:0.500000"
// FLOAT-DAG: "f32:0"
// FLOAT-DAG: "f32:1"
// FLOAT-DAG: "f32:8"
// FLOAT-DAG: "f32:2"

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
// INT-POISON-DAG: "status": "blocked"
// INT-POISON-DAG: "math.absi cannot represent absolute value of signed minimum"

module {
  dataflow.graph private @math_unary_float(%ctrl: none)
      -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
          f32, f32, f32, f32, f32, f32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 17, 0, 0>} {
    %neg = dataflow.constant %ctrl {const_value = -3.000000e+00 : f32} : f32
    %abs = math.absf %neg : f32
    %sixteen = dataflow.constant %ctrl {const_value = 1.600000e+01 : f32} : f32
    %sqrt = math.sqrt %sixteen : f32
    %four = dataflow.constant %ctrl {const_value = 4.000000e+00 : f32} : f32
    %rsqrt = math.rsqrt %four : f32
    %zero = dataflow.constant %ctrl {const_value = 0.000000e+00 : f32} : f32
    %sin = math.sin %zero : f32
    %cos = math.cos %zero : f32
    %tan = math.tan %zero : f32
    %sinh = math.sinh %zero : f32
    %cosh = math.cosh %zero : f32
    %tanh = math.tanh %zero : f32
    %exp = math.exp %zero : f32
    %three = dataflow.constant %ctrl {const_value = 3.000000e+00 : f32} : f32
    %exp2 = math.exp2 %three : f32
    %expm1 = math.expm1 %zero : f32
    %one = dataflow.constant %ctrl {const_value = 1.000000e+00 : f32} : f32
    %log = math.log %one : f32
    %eight = dataflow.constant %ctrl {const_value = 8.000000e+00 : f32} : f32
    %log2 = math.log2 %eight : f32
    %hundred = dataflow.constant %ctrl {const_value = 1.000000e+02 : f32} : f32
    %log10 = math.log10 %hundred : f32
    %log1p = math.log1p %zero : f32
    %erf = math.erf %zero : f32
    %published:18 = dataflow.sync %ctrl, %abs, %sqrt, %rsqrt, %sin, %cos,
        %tan, %sinh, %cosh, %tanh, %exp, %exp2, %expm1, %log, %log2,
        %log10, %log1p, %erf
        : (none, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
           f32, f32, f32, f32, f32, f32)
          -> (none, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
              f32, f32, f32, f32, f32, f32, f32)
    dataflow.graph.return %published#0, %published#1, %published#2,
        %published#3, %published#4, %published#5, %published#6,
        %published#7, %published#8, %published#9, %published#10,
        %published#11, %published#12, %published#13, %published#14,
        %published#15, %published#16, %published#17
        : none, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
          f32, f32, f32, f32, f32, f32
  }

  dataflow.graph private @math_rounding_float(%ctrl: none)
      -> (f32, f32, f32, f32, f32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 5, 0, 0>} {
    %floor_in = dataflow.constant %ctrl {const_value = 2.750000e+00 : f32} : f32
    %floor = math.floor %floor_in : f32
    %ceil_in = dataflow.constant %ctrl {const_value = 2.250000e+00 : f32} : f32
    %ceil = math.ceil %ceil_in : f32
    %round_in = dataflow.constant %ctrl {const_value = 2.500000e+00 : f32} : f32
    %round = math.round %round_in : f32
    %trunc_in = dataflow.constant %ctrl {const_value = -2.750000e+00 : f32} : f32
    %trunc = math.trunc %trunc_in : f32
    %even = math.roundeven %round_in : f32
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
    %neg_half = math.roundeven %neg_half_in : f32
    %pos_half_in = dataflow.constant %ctrl {const_value = 5.000000e-01 : f32} : f32
    %pos_half = math.roundeven %pos_half_in : f32
    %one_half_in = dataflow.constant %ctrl {const_value = 1.500000e+00 : f32} : f32
    %one_half = math.roundeven %one_half_in : f32
    %two_half_in = dataflow.constant %ctrl {const_value = 2.500000e+00 : f32} : f32
    %two_half = math.roundeven %two_half_in : f32
    %three_half_in = dataflow.constant %ctrl {const_value = 3.500000e+00 : f32} : f32
    %three_half = math.roundeven %three_half_in : f32
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
