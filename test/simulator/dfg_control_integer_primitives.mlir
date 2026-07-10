// RUN: loom-dfg-sim %s --graph compare_select --arg 0=none --output %t.compare.json
// RUN: FileCheck %s --check-prefix=COMPARE < %t.compare.json
// RUN: loom-dfg-sim %s --graph integer_mix --arg 0=none --output %t.integer.json
// RUN: FileCheck %s --check-prefix=INTEGER < %t.integer.json
// RUN: loom-dfg-sim %s --graph byte_swap --arg 0=none --output %t.bswap.json
// RUN: FileCheck %s --check-prefix=BSWAP < %t.bswap.json
// RUN: loom-dfg-sim %s --graph zext_bits --arg 0=none --output %t.zext.json
// RUN: FileCheck %s --check-prefix=ZEXT < %t.zext.json
// RUN: loom-dfg-sim %s --graph uint_to_float --arg 0=none --output %t.uitofp.json
// RUN: FileCheck %s --check-prefix=UITOFP < %t.uitofp.json
// RUN: loom-dfg-sim %s --graph unsigned_extend_and_minmax --arg 0=none --output %t.unsigned.json
// RUN: FileCheck %s --check-prefix=UNSIGNED < %t.unsigned.json
// RUN: loom-dfg-sim %s --graph unsigned_saturating_sub --arg 0=none --output %t.usub_sat.json
// RUN: FileCheck %s --check-prefix=USUB-SAT < %t.usub_sat.json
// RUN: loom-dfg-sim %s --graph signed_minmax --arg 0=none --output %t.signed-minmax.json
// RUN: FileCheck %s --check-prefix=SIGNED-MINMAX < %t.signed-minmax.json
// RUN: loom-dfg-sim %s --graph count_leading_zeros --arg 0=none --output %t.ctlz.json
// RUN: FileCheck %s --check-prefix=CTLZ < %t.ctlz.json

// COMPARE-DAG: "workload": "compare_select"
// COMPARE-DAG: "graph": "compare_select"
// COMPARE-DAG: "status": "pass"
// COMPARE-DAG: "operation_cost_model_source": "loom.sim.operation_cost.v1"
// COMPARE-DAG: "operation_cost_score": 8
// COMPARE-DAG: "event_count": 4
// COMPARE-DAG: "f32:3"

// INTEGER-DAG: "workload": "integer_mix"
// INTEGER-DAG: "graph": "integer_mix"
// INTEGER-DAG: "status": "pass"
// INTEGER-DAG: "operation_cost_model_source": "loom.sim.operation_cost.v1"
// INTEGER-DAG: "operation_cost_score": 29
// INTEGER-DAG: "event_count": 14
// INTEGER-DAG: "i32:3"

// BSWAP-DAG: "workload": "byte_swap"
// BSWAP-DAG: "graph": "byte_swap"
// BSWAP-DAG: "status": "pass"
// BSWAP-DAG: "operation_cost_score": 4
// BSWAP-DAG: "event_count": 2
// BSWAP-DAG: "i32:2018915346"

// ZEXT-DAG: "workload": "zext_bits"
// ZEXT-DAG: "graph": "zext_bits"
// ZEXT-DAG: "status": "pass"
// ZEXT-DAG: "operation_cost_score": 4
// ZEXT-DAG: "i64:4294967295"

// UITOFP-DAG: "workload": "uint_to_float"
// UITOFP-DAG: "graph": "uint_to_float"
// UITOFP-DAG: "status": "pass"
// UITOFP-DAG: "operation_cost_score": 6
// UITOFP-DAG: "event_count": 2
// UITOFP-DAG: "f32:7"

// UNSIGNED-DAG: "workload": "unsigned_extend_and_minmax"
// UNSIGNED-DAG: "graph": "unsigned_extend_and_minmax"
// UNSIGNED-DAG: "status": "pass"
// UNSIGNED-DAG: "arith.extui": 1
// UNSIGNED-DAG: "arith.index_castui": 2
// UNSIGNED-DAG: "llvm.intr.umin": 1
// UNSIGNED-DAG: "llvm.intr.umax": 1
// UNSIGNED-DAG: "i32:255"
// UNSIGNED-DAG: "i32:7"
// UNSIGNED-DAG: "i32:-1"
// UNSIGNED-DAG: "index:255"
// UNSIGNED-DAG: "i32:1"

// USUB-SAT-DAG: "workload": "unsigned_saturating_sub"
// USUB-SAT-DAG: "graph": "unsigned_saturating_sub"
// USUB-SAT-DAG: "status": "pass"
// USUB-SAT-DAG: "llvm.intr.usub.sat": 2
// USUB-SAT-DAG: "i32:0"
// USUB-SAT-DAG: "i32:5"

// SIGNED-MINMAX-DAG: "workload": "signed_minmax"
// SIGNED-MINMAX-DAG: "graph": "signed_minmax"
// SIGNED-MINMAX-DAG: "status": "pass"
// SIGNED-MINMAX-DAG: "llvm.intr.smin": 1
// SIGNED-MINMAX-DAG: "llvm.intr.smax": 1
// SIGNED-MINMAX-DAG: "i8:-4"
// SIGNED-MINMAX-DAG: "i8:7"

// CTLZ-DAG: "workload": "count_leading_zeros"
// CTLZ-DAG: "graph": "count_leading_zeros"
// CTLZ-DAG: "status": "pass"
// CTLZ-DAG: "llvm.intr.ctlz": 1
// CTLZ-DAG: "i32:27"

module {
  dataflow.graph.func private @compare_select(%ctrl: none) -> (none, f32) {
    %lhs = dataflow.constant %ctrl {const_value = 9.000000e+00 : f32} : f32
    %rhs = dataflow.constant %ctrl {const_value = 3.000000e+00 : f32} : f32
    %pred = arith.cmpf ugt, %lhs, %rhs : f32
    %selected = llvm.select %pred, %rhs, %lhs : i1, f32
    dataflow.graph.return %ctrl, %selected : none, f32
  }

  dataflow.graph.func private @integer_mix(%ctrl: none) -> (none, i32) {
    %wide = dataflow.constant %ctrl {const_value = 305419896 : i64} : i64
    %value = llvm.trunc %wide : i64 to i32
    %amount = dataflow.constant %ctrl {const_value = 4 : i32} : i32
    %rotated = llvm.intr.fshl(%value, %value, %amount) : (i32, i32, i32) -> i32
    %mask = dataflow.constant %ctrl {const_value = 255 : i32} : i32
    %xored = arith.xori %rotated, %mask : i32
    %modulus = dataflow.constant %ctrl {const_value = 13 : i32} : i32
    %reduced = arith.remui %xored, %modulus : i32
    %offset = dataflow.constant %ctrl {const_value = 5 : i32} : i32
    %subtracted = arith.subi %reduced, %offset : i32
    %zero = dataflow.constant %ctrl {const_value = 0 : i32} : i32
    %is_nonzero = arith.cmpi ne, %subtracted, %zero : i32
    %fallback = dataflow.constant %ctrl {const_value = 99 : i32} : i32
    %selected = arith.select %is_nonzero, %subtracted, %fallback : i32
    dataflow.graph.return %ctrl, %selected : none, i32
  }

  dataflow.graph.func private @byte_swap(%ctrl: none) -> (none, i32) {
    %value = dataflow.constant %ctrl {const_value = 305419896 : i32} : i32
    %swapped = llvm.intr.bswap(%value) : (i32) -> i32
    dataflow.graph.return %ctrl, %swapped : none, i32
  }

  dataflow.graph.func private @zext_bits(%ctrl: none) -> (none, i64) {
    %value = dataflow.constant %ctrl {const_value = -1 : i32} : i32
    %wide = llvm.zext %value : i32 to i64
    dataflow.graph.return %ctrl, %wide : none, i64
  }

  dataflow.graph.func private @uint_to_float(%ctrl: none) -> (none, f32) {
    %value = dataflow.constant %ctrl {const_value = 7 : i32} : i32
    %fp = llvm.uitofp %value : i32 to f32
    dataflow.graph.return %ctrl, %fp : none, f32
  }

  dataflow.graph.func private @unsigned_extend_and_minmax(%ctrl: none)
      -> (none, i32, i32, i32, index, i32) {
    %byte = dataflow.constant %ctrl {const_value = -1 : i8} : i8
    %wide = arith.extui %byte : i8 to i32
    %idx = arith.index_castui %byte : i8 to index
    %wide_idx = dataflow.constant %ctrl {const_value = 4294967297 : index} : index
    %narrow_idx = arith.index_castui %wide_idx : index to i32
    %seven = dataflow.constant %ctrl {const_value = 7 : i32} : i32
    %minus_one = dataflow.constant %ctrl {const_value = -1 : i32} : i32
    %min = llvm.intr.umin(%minus_one, %seven) : (i32, i32) -> i32
    %max = llvm.intr.umax(%minus_one, %seven) : (i32, i32) -> i32
    dataflow.graph.return %ctrl, %wide, %min, %max, %idx, %narrow_idx
        : none, i32, i32, i32, index, i32
  }

  dataflow.graph.func private @unsigned_saturating_sub(%ctrl: none)
      -> (none, i32, i32) {
    %small = dataflow.constant %ctrl {const_value = 3 : i32} : i32
    %large = dataflow.constant %ctrl {const_value = 5 : i32} : i32
    %underflow = llvm.intr.usub.sat(%small, %large) : (i32, i32) -> i32
    %nine = dataflow.constant %ctrl {const_value = 9 : i32} : i32
    %four = dataflow.constant %ctrl {const_value = 4 : i32} : i32
    %difference = llvm.intr.usub.sat(%nine, %four) : (i32, i32) -> i32
    dataflow.graph.return %ctrl, %underflow, %difference : none, i32, i32
  }

  dataflow.graph.func private @signed_minmax(%ctrl: none)
      -> (none, i8, i8) {
    %minus_four = dataflow.constant %ctrl {const_value = -4 : i8} : i8
    %seven = dataflow.constant %ctrl {const_value = 7 : i8} : i8
    %min = llvm.intr.smin(%minus_four, %seven) : (i8, i8) -> i8
    %max = llvm.intr.smax(%minus_four, %seven) : (i8, i8) -> i8
    dataflow.graph.return %ctrl, %min, %max : none, i8, i8
  }

  dataflow.graph.func private @count_leading_zeros(%ctrl: none)
      -> (none, i32) {
    %value = dataflow.constant %ctrl {const_value = 16 : i32} : i32
    %zeros = "llvm.intr.ctlz"(%value) <{is_zero_poison = false}> : (i32) -> i32
    dataflow.graph.return %ctrl, %zeros : none, i32
  }
}
