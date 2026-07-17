// RUN: loom-dfg-sim %s --graph signed_shift_div_rem --output %t.signed.json
// RUN: FileCheck %s --check-prefix=SIGNED < %t.signed.json
// RUN: loom-dfg-sim %s --graph unsigned_division --output %t.unsigned-div.json
// RUN: FileCheck %s --check-prefix=UNSIGNED-DIV < %t.unsigned-div.json
// RUN: loom-dfg-sim %s --graph extend_truncate --output %t.cast.json
// RUN: FileCheck %s --check-prefix=CAST < %t.cast.json
// RUN: loom-dfg-sim %s --graph narrow_runtime_signed_compare --arg 0=255 --arg 1=0 --arg 2=-1 --output %t.narrow-cmp.json
// RUN: FileCheck %s --check-prefix=NARROW-CMP < %t.narrow-cmp.json
// RUN: loom-dfg-sim %s --graph i64_wraparound --output %t.i64-wrap.json
// RUN: FileCheck %s --check-prefix=I64-WRAP < %t.i64-wrap.json
// RUN: loom-dfg-sim %s --graph llvm_sign_extend --output %t.llvm-sext.json
// RUN: FileCheck %s --check-prefix=LLVM-SEXT < %t.llvm-sext.json
// RUN: loom-dfg-sim %s --graph exact_division_poison --output %t.exact-div.json
// RUN: FileCheck %s --check-prefix=EXACT-DIV < %t.exact-div.json
// RUN: loom-dfg-sim %s --graph exact_shift_poison --output %t.exact-shift.json
// RUN: FileCheck %s --check-prefix=EXACT-SHIFT < %t.exact-shift.json
// RUN: loom-dfg-sim %s --graph exact_unsigned_shift_poison --output %t.exact-unsigned-shift.json
// RUN: FileCheck %s --check-prefix=EXACT-UNSIGNED-SHIFT < %t.exact-unsigned-shift.json
// RUN: loom-dfg-sim %s --graph oversized_shift_poison --output %t.oversized-shift.json
// RUN: FileCheck %s --check-prefix=OVERSIZED-SHIFT < %t.oversized-shift.json
// RUN: loom-dfg-sim %s --graph remsi_min_minus_one --output %t.remsi-edge.json
// RUN: FileCheck %s --check-prefix=REMSI-EDGE < %t.remsi-edge.json
// RUN: loom-dfg-sim %s --graph trunci_overflow_poison --output %t.trunci-overflow.json
// RUN: FileCheck %s --check-prefix=TRUNCI-OVERFLOW < %t.trunci-overflow.json

// SIGNED-DAG: "workload": "signed_shift_div_rem"
// SIGNED-DAG: "graph": "signed_shift_div_rem"
// SIGNED-DAG: "status": "pass"
// SIGNED-DAG: "operation_cost_score": 28
// SIGNED-DAG: "event_count": 8
// SIGNED-DAG: "arith.shrsi": 1
// SIGNED-DAG: "arith.divsi": 1
// SIGNED-DAG: "arith.remsi": 1
// SIGNED-DAG: "i32:-5"

// UNSIGNED-DIV-DAG: "workload": "unsigned_division"
// UNSIGNED-DIV-DAG: "graph": "unsigned_division"
// UNSIGNED-DIV-DAG: "status": "pass"
// UNSIGNED-DIV-DAG: "arith.divui": 1
// UNSIGNED-DIV-DAG: "i32:1431655765"

// CAST-DAG: "workload": "extend_truncate"
// CAST-DAG: "graph": "extend_truncate"
// CAST-DAG: "status": "pass"
// CAST-DAG: "operation_cost_score": 9
// CAST-DAG: "event_count": 5
// CAST-DAG: "arith.extsi": 1
// CAST-DAG: "arith.trunci": 1
// CAST-DAG: "i32:-2"
// CAST-DAG: "i8:52"

// NARROW-CMP-DAG: "workload": "narrow_runtime_signed_compare"
// NARROW-CMP-DAG: "graph": "narrow_runtime_signed_compare"
// NARROW-CMP-DAG: "status": "pass"
// NARROW-CMP-DAG: "arith.cmpi": 5
// NARROW-CMP-DAG: "i1:true"
// NARROW-CMP-DAG: "i1:true"
// NARROW-CMP-DAG: "i1:true"
// NARROW-CMP-DAG: "i1:false"
// NARROW-CMP-DAG: "i1:true"

// I64-WRAP: "final_outputs": [
// I64-WRAP-NEXT: "none",
// I64-WRAP-NEXT: "i64:-9223372036854775808",
// I64-WRAP-NEXT: "i64:9223372036854775807",
// I64-WRAP-NEXT: "i64:-9223372036854775808"
// I64-WRAP-DAG: "arith.addi": 1
// I64-WRAP-DAG: "arith.subi": 1
// I64-WRAP-DAG: "arith.muli": 1
// I64-WRAP-DAG: "status": "pass"
// I64-WRAP-DAG: "workload": "i64_wraparound"

// LLVM-SEXT-DAG: "workload": "llvm_sign_extend"
// LLVM-SEXT-DAG: "graph": "llvm_sign_extend"
// LLVM-SEXT-DAG: "status": "pass"
// LLVM-SEXT-DAG: "llvm.sext": 1
// LLVM-SEXT-DAG: "i32:-2"

// EXACT-DIV-DAG: "workload": "exact_division_poison"
// EXACT-DIV-DAG: "status": "blocked"
// EXACT-DIV-DAG: "arith.divsi exact result would be poison"

// EXACT-SHIFT-DAG: "workload": "exact_shift_poison"
// EXACT-SHIFT-DAG: "status": "blocked"
// EXACT-SHIFT-DAG: "arith.shrsi exact shift would discard non-zero bits"

// EXACT-UNSIGNED-SHIFT-DAG: "workload": "exact_unsigned_shift_poison"
// EXACT-UNSIGNED-SHIFT-DAG: "status": "blocked"
// EXACT-UNSIGNED-SHIFT-DAG: "arith.shrui exact shift would discard non-zero bits"

// OVERSIZED-SHIFT-DAG: "workload": "oversized_shift_poison"
// OVERSIZED-SHIFT-DAG: "status": "blocked"
// OVERSIZED-SHIFT-DAG: "arith.shrsi shift amount must be less than bit width 8, got 8"

// REMSI-EDGE-DAG: "workload": "remsi_min_minus_one"
// REMSI-EDGE-DAG: "status": "pass"
// REMSI-EDGE-DAG: "i8:0"

// TRUNCI-OVERFLOW-DAG: "workload": "trunci_overflow_poison"
// TRUNCI-OVERFLOW-DAG: "status": "blocked"
// TRUNCI-OVERFLOW-DAG: "arith.trunci overflow<nuw> result would be poison"
// TRUNCI-OVERFLOW-DAG: "arith.trunci overflow<nsw> result would be poison"

module {
  dataflow.graph.func private @signed_shift_div_rem(%ctrl: none) -> (none, i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %negative = dataflow.constant %ctrl {const_value = -33 : i32} : i32
    %amount = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    %shifted = arith.shrsi %negative, %amount : i32
    %divisor = dataflow.constant %ctrl {const_value = 5 : i32} : i32
    %quotient = arith.divsi %shifted, %divisor : i32
    %remainder = arith.remsi %shifted, %divisor : i32
    %combined = arith.addi %quotient, %remainder : i32
    %published:2 = dataflow.sync %ctrl, %combined
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }

  dataflow.graph.func private @unsigned_division(%ctrl: none) -> (none, i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %wide = dataflow.constant %ctrl {const_value = -1 : i32} : i32
    %divisor = dataflow.constant %ctrl {const_value = 3 : i32} : i32
    %quotient = arith.divui %wide, %divisor : i32
    %published:2 = dataflow.sync %ctrl, %quotient
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }

  dataflow.graph.func private @extend_truncate(%ctrl: none) -> (none, i32, i8)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 2, 0, 0>} {
    %byte = dataflow.constant %ctrl {const_value = -2 : i8} : i8
    %wide = arith.extsi %byte : i8 to i32
    %base = dataflow.constant %ctrl {const_value = 4660 : i32} : i32
    %narrow = arith.trunci %base : i32 to i8
    %published:3 = dataflow.sync %ctrl, %wide, %narrow
        : (none, i32, i8) -> (none, i32, i8)
    dataflow.graph.return %published#0, %published#1, %published#2
        : none, i32, i8
  }

  dataflow.graph.func private @narrow_runtime_signed_compare(
      %ctrl: none, %lhs: i8, %zero: i8, %minus_one: i8)
      -> (none, i1, i1, i1, i1, i1)
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 5, 0, 0>} {
    %slt = arith.cmpi slt, %lhs, %zero : i8
    %sgt = arith.cmpi sgt, %zero, %lhs : i8
    %eq = arith.cmpi eq, %lhs, %minus_one : i8
    %ult = arith.cmpi ult, %lhs, %zero : i8
    %ugt = arith.cmpi ugt, %lhs, %zero : i8
    %published:6 = dataflow.sync %ctrl, %slt, %sgt, %eq, %ult, %ugt
        : (none, i1, i1, i1, i1, i1) -> (none, i1, i1, i1, i1, i1)
    dataflow.graph.return %published#0, %published#1, %published#2,
        %published#3, %published#4, %published#5
        : none, i1, i1, i1, i1, i1
  }

  dataflow.graph.func private @i64_wraparound(%ctrl: none)
      -> (none, i64, i64, i64)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 3, 0, 0>} {
    %max = dataflow.constant %ctrl
        {const_value = 9223372036854775807 : i64} : i64
    %min = dataflow.constant %ctrl
        {const_value = -9223372036854775808 : i64} : i64
    %one = dataflow.constant %ctrl {const_value = 1 : i64} : i64
    %minus_one = dataflow.constant %ctrl {const_value = -1 : i64} : i64
    %add = arith.addi %max, %one : i64
    %sub = arith.subi %min, %one : i64
    %mul = arith.muli %min, %minus_one : i64
    %published:4 = dataflow.sync %ctrl, %add, %sub, %mul
        : (none, i64, i64, i64) -> (none, i64, i64, i64)
    dataflow.graph.return %published#0, %published#1, %published#2,
        %published#3 : none, i64, i64, i64
  }

  dataflow.graph.func private @llvm_sign_extend(%ctrl: none) -> (none, i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %byte = dataflow.constant %ctrl {const_value = -2 : i8} : i8
    %wide = llvm.sext %byte : i8 to i32
    %published:2 = dataflow.sync %ctrl, %wide
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }

  dataflow.graph.func private @exact_division_poison(%ctrl: none) -> (none, i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %lhs = dataflow.constant %ctrl {const_value = 5 : i32} : i32
    %rhs = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    %quotient = arith.divsi %lhs, %rhs exact : i32
    %published:2 = dataflow.sync %ctrl, %quotient
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }

  dataflow.graph.func private @exact_shift_poison(%ctrl: none) -> (none, i8)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %ctrl {const_value = -3 : i8} : i8
    %amount = dataflow.constant %ctrl {const_value = 1 : i8} : i8
    %shifted = arith.shrsi %value, %amount exact : i8
    %published:2 = dataflow.sync %ctrl, %shifted
        : (none, i8) -> (none, i8)
    dataflow.graph.return %published#0, %published#1 : none, i8
  }

  dataflow.graph.func private @exact_unsigned_shift_poison(%ctrl: none)
      -> (none, i8)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %ctrl {const_value = 3 : i8} : i8
    %amount = dataflow.constant %ctrl {const_value = 1 : i8} : i8
    %shifted = arith.shrui %value, %amount exact : i8
    %published:2 = dataflow.sync %ctrl, %shifted
        : (none, i8) -> (none, i8)
    dataflow.graph.return %published#0, %published#1 : none, i8
  }

  dataflow.graph.func private @oversized_shift_poison(%ctrl: none)
      -> (none, i8)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %ctrl {const_value = -2 : i8} : i8
    %amount = dataflow.constant %ctrl {const_value = 8 : i8} : i8
    %shifted = arith.shrsi %value, %amount : i8
    %published:2 = dataflow.sync %ctrl, %shifted
        : (none, i8) -> (none, i8)
    dataflow.graph.return %published#0, %published#1 : none, i8
  }

  dataflow.graph.func private @remsi_min_minus_one(%ctrl: none) -> (none, i8)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %min = dataflow.constant %ctrl {const_value = -128 : i8} : i8
    %minus_one = dataflow.constant %ctrl {const_value = -1 : i8} : i8
    %remainder = arith.remsi %min, %minus_one : i8
    %published:2 = dataflow.sync %ctrl, %remainder
        : (none, i8) -> (none, i8)
    dataflow.graph.return %published#0, %published#1 : none, i8
  }

  dataflow.graph.func private @trunci_overflow_poison(%ctrl: none)
      -> (none, i8, i8)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 2, 0, 0>} {
    %nuw_source = dataflow.constant %ctrl {const_value = 256 : i32} : i32
    %nuw = arith.trunci %nuw_source overflow<nuw> : i32 to i8
    %nsw_source = dataflow.constant %ctrl {const_value = 128 : i32} : i32
    %nsw = arith.trunci %nsw_source overflow<nsw> : i32 to i8
    %published:3 = dataflow.sync %ctrl, %nuw, %nsw
        : (none, i8, i8) -> (none, i8, i8)
    dataflow.graph.return %published#0, %published#1, %published#2
        : none, i8, i8
  }
}
