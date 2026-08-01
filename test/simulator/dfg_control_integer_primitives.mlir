// RUN: loom-dfg-sim %s --graph compare_select --output %t.compare.json
// RUN: FileCheck %s --check-prefix=COMPARE < %t.compare.json
// RUN: loom-dfg-sim %s --graph integer_mix --output %t.integer.json
// RUN: FileCheck %s --check-prefix=INTEGER < %t.integer.json
// RUN: loom-dfg-sim %s --graph byte_swap --output %t.bswap.json
// RUN: FileCheck %s --check-prefix=BSWAP < %t.bswap.json
// RUN: loom-dfg-sim %s --graph zext_bits --output %t.zext.json
// RUN: FileCheck %s --check-prefix=ZEXT < %t.zext.json
// RUN: loom-dfg-sim %s --graph uint_to_float --output %t.uitofp.json
// RUN: FileCheck %s --check-prefix=UITOFP < %t.uitofp.json
// RUN: loom-dfg-sim %s --graph unsigned_extend_and_minmax --output %t.unsigned.json
// RUN: FileCheck %s --check-prefix=UNSIGNED < %t.unsigned.json
// RUN: loom-dfg-sim %s --graph unsigned_saturating_sub --output %t.usub_sat.json
// RUN: FileCheck %s --check-prefix=USUB-SAT < %t.usub_sat.json
// RUN: loom-dfg-sim %s --graph signed_minmax --output %t.signed-minmax.json
// RUN: FileCheck %s --check-prefix=SIGNED-MINMAX < %t.signed-minmax.json
// RUN: loom-dfg-sim %s --graph count_leading_zeros --output %t.ctlz.json
// RUN: FileCheck %s --check-prefix=CTLZ < %t.ctlz.json
// RUN: loom-dfg-sim %s --graph bitcast_roundtrip --output %t.bitcast.json
// RUN: FileCheck %s --check-prefix=BITCAST < %t.bitcast.json

// COMPARE-DAG: "workload": "compare_select"
// COMPARE-DAG: "graph": "compare_select"
// COMPARE-DAG: "status": "pass"
// COMPARE-DAG: "event_count": 5
// COMPARE-DAG: "f32:3"

// INTEGER-DAG: "workload": "integer_mix"
// INTEGER-DAG: "graph": "integer_mix"
// INTEGER-DAG: "status": "pass"
// INTEGER-DAG: "event_count": 15
// INTEGER-DAG: "i32:3"

// BSWAP-DAG: "workload": "byte_swap"
// BSWAP-DAG: "graph": "byte_swap"
// BSWAP-DAG: "status": "pass"
// BSWAP-DAG: "event_count": 3
// BSWAP-DAG: "i32:2018915346"

// ZEXT-DAG: "workload": "zext_bits"
// ZEXT-DAG: "graph": "zext_bits"
// ZEXT-DAG: "status": "pass"
// ZEXT-DAG: "i64:4294967295"

// UITOFP-DAG: "workload": "uint_to_float"
// UITOFP-DAG: "graph": "uint_to_float"
// UITOFP-DAG: "status": "pass"
// UITOFP-DAG: "event_count": 3
// UITOFP-DAG: "f32:7"

// UNSIGNED-DAG: "workload": "unsigned_extend_and_minmax"
// UNSIGNED-DAG: "graph": "unsigned_extend_and_minmax"
// UNSIGNED-DAG: "status": "pass"
// UNSIGNED-DAG: "arith.extui": 1
// UNSIGNED-DAG: "arith.index_castui": 2
// UNSIGNED-DAG: "arith.minui": 1
// UNSIGNED-DAG: "arith.maxui": 1
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
// SIGNED-MINMAX-DAG: "arith.minsi": 1
// SIGNED-MINMAX-DAG: "arith.maxsi": 1
// SIGNED-MINMAX-DAG: "i8:-4"
// SIGNED-MINMAX-DAG: "i8:7"

// CTLZ-DAG: "workload": "count_leading_zeros"
// CTLZ-DAG: "graph": "count_leading_zeros"
// CTLZ-DAG: "status": "pass"
// CTLZ-DAG: "math.ctlz": 1
// CTLZ-DAG: "i32:27"

// BITCAST-DAG: "workload": "bitcast_roundtrip"
// BITCAST-DAG: "graph": "bitcast_roundtrip"
// BITCAST-DAG: "status": "pass"
// BITCAST-DAG: "arith.bitcast": 2
// BITCAST-DAG: "i16:15360"

module {
  dataflow.graph private @bitcast_roundtrip(%ctrl: none) -> (i16)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %bits = dataflow.constant %ctrl {const_value = 15360 : i16} : i16
    %as_float = arith.bitcast %bits : i16 to f16
    %roundtrip = arith.bitcast %as_float : f16 to i16
    %published:2 = dataflow.sync %ctrl, %roundtrip
        : (none, i16) -> (none, i16)
    dataflow.graph.return %published#0, %published#1 : none, i16
  }

  dataflow.graph private @compare_select(%ctrl: none) -> (f32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %lhs = dataflow.constant %ctrl {const_value = 9.000000e+00 : f32} : f32
    %rhs = dataflow.constant %ctrl {const_value = 3.000000e+00 : f32} : f32
    %pred = arith.cmpf ugt, %lhs, %rhs : f32
    %selected = arith.select %pred, %rhs, %lhs : f32
    %published:2 = dataflow.sync %ctrl, %selected
        : (none, f32) -> (none, f32)
    dataflow.graph.return %published#0, %published#1 : none, f32
  }

  dataflow.graph private @integer_mix(%ctrl: none) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %wide = dataflow.constant %ctrl {const_value = 305419896 : i64} : i64
    %value = arith.trunci %wide : i64 to i32
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
    %published:2 = dataflow.sync %ctrl, %selected
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }

  dataflow.graph private @byte_swap(%ctrl: none) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %ctrl {const_value = 305419896 : i32} : i32
    %swapped = llvm.intr.bswap(%value) : (i32) -> i32
    %published:2 = dataflow.sync %ctrl, %swapped
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }

  dataflow.graph private @zext_bits(%ctrl: none) -> (i64)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %ctrl {const_value = -1 : i32} : i32
    %wide = arith.extui %value : i32 to i64
    %published:2 = dataflow.sync %ctrl, %wide
        : (none, i64) -> (none, i64)
    dataflow.graph.return %published#0, %published#1 : none, i64
  }

  dataflow.graph private @uint_to_float(%ctrl: none) -> (f32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %ctrl {const_value = 7 : i32} : i32
    %fp = arith.uitofp %value : i32 to f32
    %published:2 = dataflow.sync %ctrl, %fp
        : (none, f32) -> (none, f32)
    dataflow.graph.return %published#0, %published#1 : none, f32
  }

  dataflow.graph private @unsigned_extend_and_minmax(%ctrl: none)
      -> (i32, i32, i32, index, i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 5, 0, 0>} {
    %byte = dataflow.constant %ctrl {const_value = -1 : i8} : i8
    %wide = arith.extui %byte : i8 to i32
    %idx = arith.index_castui %byte : i8 to index
    %wide_idx = dataflow.constant %ctrl {const_value = 4294967297 : index} : index
    %narrow_idx = arith.index_castui %wide_idx : index to i32
    %seven = dataflow.constant %ctrl {const_value = 7 : i32} : i32
    %minus_one = dataflow.constant %ctrl {const_value = -1 : i32} : i32
    %min = arith.minui %minus_one, %seven : i32
    %max = arith.maxui %minus_one, %seven : i32
    %published:6 = dataflow.sync %ctrl, %wide, %min, %max, %idx, %narrow_idx
        : (none, i32, i32, i32, index, i32)
          -> (none, i32, i32, i32, index, i32)
    dataflow.graph.return %published#0, %published#1, %published#2,
        %published#3, %published#4, %published#5
        : none, i32, i32, i32, index, i32
  }

  dataflow.graph private @unsigned_saturating_sub(%ctrl: none)
      -> (i32, i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 2, 0, 0>} {
    %small = dataflow.constant %ctrl {const_value = 3 : i32} : i32
    %large = dataflow.constant %ctrl {const_value = 5 : i32} : i32
    %underflow = llvm.intr.usub.sat(%small, %large) : (i32, i32) -> i32
    %nine = dataflow.constant %ctrl {const_value = 9 : i32} : i32
    %four = dataflow.constant %ctrl {const_value = 4 : i32} : i32
    %difference = llvm.intr.usub.sat(%nine, %four) : (i32, i32) -> i32
    %published:3 = dataflow.sync %ctrl, %underflow, %difference
        : (none, i32, i32) -> (none, i32, i32)
    dataflow.graph.return %published#0, %published#1, %published#2
        : none, i32, i32
  }

  dataflow.graph private @signed_minmax(%ctrl: none)
      -> (i8, i8)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 2, 0, 0>} {
    %minus_four = dataflow.constant %ctrl {const_value = -4 : i8} : i8
    %seven = dataflow.constant %ctrl {const_value = 7 : i8} : i8
    %min = arith.minsi %minus_four, %seven : i8
    %max = arith.maxsi %minus_four, %seven : i8
    %published:3 = dataflow.sync %ctrl, %min, %max
        : (none, i8, i8) -> (none, i8, i8)
    dataflow.graph.return %published#0, %published#1, %published#2
        : none, i8, i8
  }

  dataflow.graph private @count_leading_zeros(%ctrl: none)
      -> (i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %ctrl {const_value = 16 : i32} : i32
    %zeros = math.ctlz %value : i32
    %published:2 = dataflow.sync %ctrl, %zeros
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}
