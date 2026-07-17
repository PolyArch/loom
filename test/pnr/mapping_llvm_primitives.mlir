// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph zext_graph --hardware-mlir %s --hardware llvm_primitive_adg --workload zext_graph --output %t.zext.csv --artifact %t.zext.json
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph sext_graph --hardware-mlir %s --hardware llvm_primitive_adg --workload sext_graph --output %t.sext.csv --artifact %t.sext.json
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph abs_graph --hardware-mlir %s --hardware llvm_primitive_adg --workload abs_graph --output %t.abs.csv --artifact %t.abs.json
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph fabs_graph --hardware-mlir %s --hardware llvm_primitive_adg --workload fabs_graph --output %t.fabs.csv --artifact %t.fabs.json
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph fmuladd_graph --hardware-mlir %s --hardware llvm_primitive_adg --workload fmuladd_graph --output %t.fmuladd.csv --artifact %t.fmuladd.json
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph fshl_graph --hardware-mlir %s --hardware llvm_primitive_adg --workload fshl_graph --output %t.fshl.csv --artifact %t.fshl.json
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph bswap_graph --hardware-mlir %s --hardware llvm_primitive_adg --workload bswap_graph --output %t.bswap.csv --artifact %t.bswap.json
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph umax_graph --hardware-mlir %s --hardware llvm_primitive_adg --workload umax_graph --output %t.umax.csv --artifact %t.umax.json
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph smin_graph --hardware-mlir %s --hardware llvm_primitive_adg --workload smin_graph --output %t.smin.csv --artifact %t.smin.json
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph smax_graph --hardware-mlir %s --hardware llvm_primitive_adg --workload smax_graph --output %t.smax.csv --artifact %t.smax.json
// RUN: FileCheck %s --check-prefix=ZEXT < %t.zext.csv
// RUN: FileCheck %s --check-prefix=SEXT < %t.sext.csv
// RUN: FileCheck %s --check-prefix=ABS < %t.abs.csv
// RUN: FileCheck %s --check-prefix=FABS < %t.fabs.csv
// RUN: FileCheck %s --check-prefix=FMULADD < %t.fmuladd.csv
// RUN: FileCheck %s --check-prefix=FSHL < %t.fshl.csv
// RUN: FileCheck %s --check-prefix=BSWAP < %t.bswap.csv
// RUN: FileCheck %s --check-prefix=UMAX < %t.umax.csv
// RUN: FileCheck %s --check-prefix=SMIN < %t.smin.csv
// RUN: FileCheck %s --check-prefix=SMAX < %t.smax.csv

// ZEXT: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// ZEXT-NEXT: zext_graph,llvm_primitive_adg,zext_graph__zext_graph__llvm_primitive_adg,2,1,0,0,pass

// SEXT: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// SEXT-NEXT: sext_graph,llvm_primitive_adg,sext_graph__sext_graph__llvm_primitive_adg,2,1,0,0,pass

// ABS: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// ABS-NEXT: abs_graph,llvm_primitive_adg,abs_graph__abs_graph__llvm_primitive_adg,2,1,0,0,pass

// FABS: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// FABS-NEXT: fabs_graph,llvm_primitive_adg,fabs_graph__fabs_graph__llvm_primitive_adg,2,1,0,0,pass

// FMULADD: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// FMULADD-NEXT: fmuladd_graph,llvm_primitive_adg,fmuladd_graph__fmuladd_graph__llvm_primitive_adg,2,1,0,0,pass

// FSHL: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// FSHL-NEXT: fshl_graph,llvm_primitive_adg,fshl_graph__fshl_graph__llvm_primitive_adg,2,1,0,0,pass

// BSWAP: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// BSWAP-NEXT: bswap_graph,llvm_primitive_adg,bswap_graph__bswap_graph__llvm_primitive_adg,2,1,0,0,pass

// UMAX: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// UMAX-NEXT: umax_graph,llvm_primitive_adg,umax_graph__umax_graph__llvm_primitive_adg,2,1,0,0,pass

// SMIN: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// SMIN-NEXT: smin_graph,llvm_primitive_adg,smin_graph__smin_graph__llvm_primitive_adg,2,1,0,0,pass

// SMAX: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// SMAX-NEXT: smax_graph,llvm_primitive_adg,smax_graph__smax_graph__llvm_primitive_adg,2,1,0,0,pass

module {
  dataflow.graph.func private @zext_graph(%ctrl: none, %narrow: i32)
      -> (none, i64) {
    %wide = llvm.zext %narrow : i32 to i64
    dataflow.graph.return %ctrl, %wide : none, i64
  }

  dataflow.graph.func private @sext_graph(%ctrl: none, %narrow: i16)
      -> (none, i32) {
    %wide = llvm.sext %narrow : i16 to i32
    dataflow.graph.return %ctrl, %wide : none, i32
  }

  dataflow.graph.func private @abs_graph(%ctrl: none, %value: i32)
      -> (none, i32) {
    %abs = "llvm.intr.abs"(%value) <{is_int_min_poison = true}> : (i32) -> i32
    dataflow.graph.return %ctrl, %abs : none, i32
  }

  dataflow.graph.func private @fabs_graph(%ctrl: none, %value: f32)
      -> (none, f32) {
    %abs = llvm.intr.fabs(%value) : (f32) -> f32
    dataflow.graph.return %ctrl, %abs : none, f32
  }

  dataflow.graph.func private @fmuladd_graph(%ctrl: none, %lhs: f32,
                                             %rhs: f32, %acc: f32)
      -> (none, f32) {
    %result = llvm.intr.fmuladd(%lhs, %rhs, %acc) : (f32, f32, f32) -> f32
    dataflow.graph.return %ctrl, %result : none, f32
  }

  dataflow.graph.func private @fshl_graph(%ctrl: none, %lhs: i32,
                                          %rhs: i32, %amount: i32)
      -> (none, i32) {
    %result = llvm.intr.fshl(%lhs, %rhs, %amount) : (i32, i32, i32) -> i32
    dataflow.graph.return %ctrl, %result : none, i32
  }

  dataflow.graph.func private @bswap_graph(%ctrl: none, %value: i32)
      -> (none, i32) {
    %result = llvm.intr.bswap(%value) : (i32) -> i32
    dataflow.graph.return %ctrl, %result : none, i32
  }

  dataflow.graph.func private @umax_graph(%ctrl: none, %lhs: i32, %rhs: i32)
      -> (none, i32) {
    %result = llvm.intr.umax(%lhs, %rhs) : (i32, i32) -> i32
    dataflow.graph.return %ctrl, %result : none, i32
  }

  dataflow.graph.func private @smin_graph(%ctrl: none, %lhs: i8, %rhs: i8)
      -> (none, i8) {
    %result = llvm.intr.smin(%lhs, %rhs) : (i8, i8) -> i8
    dataflow.graph.return %ctrl, %result : none, i8
  }

  dataflow.graph.func private @smax_graph(%ctrl: none, %lhs: i8, %rhs: i8)
      -> (none, i8) {
    %result = llvm.intr.smax(%lhs, %rhs) : (i8, i8) -> i8
    dataflow.graph.return %ctrl, %result : none, i8
  }

  fabric.module @llvm_primitive_adg(%ctrl : !fabric.bits<0>,
                                    %i32a : !fabric.bits<32>,
                                    %i32b : !fabric.bits<32>,
                                    %i32c : !fabric.bits<32>) {
    %a_zext, %a_sext, %a_abs, %a_fabs, %a_fmuladd, %a_fshl, %a_bswap,
        %a_umax, %a_smin, %a_smax = fabric.switch [spatial] %i32a
          [{connectivity_table = ["1", "1", "1", "1", "1", "1", "1", "1",
                                  "1", "1"]}]
          : (!fabric.bits<32>)
            -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                !fabric.bits<32>)
    %b_fmuladd, %b_fshl, %b_umax, %b_smin, %b_smax =
        fabric.switch [spatial] %i32b
          [{connectivity_table = ["1", "1", "1", "1", "1"]}]
          : (!fabric.bits<32>)
            -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                !fabric.bits<32>, !fabric.bits<32>)
    %c_fmuladd, %c_fshl = fabric.switch [spatial] %i32c
        [{connectivity_table = ["1", "1"]}]
        : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
    %zext_out = fabric.pe [spatial]
        (%pa = %a_zext : !fabric.bits<32> to !fabric.bits<64>)
        -> !fabric.bits<64> {
      fabric.fu(%value = %pa : !fabric.bits<64> to !fabric.bits<32>)
          -> !fabric.bits<64> {
        %wide = fabric.op [@llvm.zext] (%value)
                : (!fabric.bits<32>) -> !fabric.bits<64>
        fabric.yield %wide : !fabric.bits<64>
      }
    }
    %sext_out = fabric.pe [spatial] (%pa = %a_sext : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
        %wide = fabric.op [@llvm.sext] (%value)
                : (!fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %wide : !fabric.bits<32>
      }
    }
    %abs_out = fabric.pe [spatial] (%pa = %a_abs : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
        %abs = fabric.op [@llvm.intr.abs] (%value)
               : (!fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %abs : !fabric.bits<32>
      }
    }
    %fabs_out = fabric.pe [spatial] (%pa = %a_fabs : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
        %abs = fabric.op [@llvm.intr.fabs] (%value)
               : (!fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %abs : !fabric.bits<32>
      }
    }
    %fmuladd_out = fabric.pe [spatial] (%pa = %a_fmuladd : !fabric.bits<32>,
                         %pb = %b_fmuladd : !fabric.bits<32>,
                         %pc = %c_fmuladd : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>,
                %acc = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
        %result = fabric.op [@llvm.intr.fmuladd] (%lhs, %rhs, %acc)
                  : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
                    -> !fabric.bits<32>
        fabric.yield %result : !fabric.bits<32>
      }
    }
    %fshl_out = fabric.pe [spatial] (%pa = %a_fshl : !fabric.bits<32>,
                         %pb = %b_fshl : !fabric.bits<32>,
                         %pc = %c_fshl : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>,
                %amount = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
        %result = fabric.op [@llvm.intr.fshl] (%lhs, %rhs, %amount)
                  : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
                    -> !fabric.bits<32>
        fabric.yield %result : !fabric.bits<32>
      }
    }
    %bswap_out = fabric.pe [spatial] (%pa = %a_bswap : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
        %result = fabric.op [@llvm.intr.bswap] (%value)
                  : (!fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %result : !fabric.bits<32>
      }
    }
    %umax_out = fabric.pe [spatial] (%pa = %a_umax : !fabric.bits<32>,
                         %pb = %b_umax : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
        %result = fabric.op [@llvm.intr.umax] (%lhs, %rhs)
                  : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %result : !fabric.bits<32>
      }
    }
    %smin_out = fabric.pe [spatial] (%pa = %a_smin : !fabric.bits<32>,
                         %pb = %b_smin : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
        %result = fabric.op [@llvm.intr.smin] (%lhs, %rhs)
                  : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %result : !fabric.bits<32>
      }
    }
    %smax_out = fabric.pe [spatial] (%pa = %a_smax : !fabric.bits<32>,
                         %pb = %b_smax : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
        %result = fabric.op [@llvm.intr.smax] (%lhs, %rhs)
                  : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %result : !fabric.bits<32>
      }
    }
    %publish_i32 = fabric.switch [spatial]
        %sext_out, %abs_out, %fabs_out, %fmuladd_out, %fshl_out,
        %bswap_out, %umax_out
        [{connectivity_table = ["1111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
           !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
           !fabric.bits<32>) -> !fabric.bits<32>
    %publish_i8 = fabric.switch [spatial] %smin_out, %smax_out
        [{connectivity_table = ["11"]}]
        : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
    %ctrl_i64, %ctrl_i32, %ctrl_i8 = fabric.switch [spatial] %ctrl
        [{connectivity_table = ["1", "1", "1"]}]
        : (!fabric.bits<0>)
          -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
    fabric.pe [spatial] (
        %pc = %ctrl_i64 : !fabric.bits<0> to !fabric.bits<64>,
        %pv = %zext_out : !fabric.bits<64>)
        -> !fabric.bits<64> {
      fabric.fu(
          %token = %pc : !fabric.bits<64> to !fabric.bits<0>,
          %value = %pv : !fabric.bits<64>) -> !fabric.bits<64> {
        %done, %published = fabric.op [@dataflow.sync] (%token, %value)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<64>)
              -> (!fabric.bits<0>, !fabric.bits<64>)
        fabric.yield %published : !fabric.bits<64>
      }
    }
    fabric.pe [spatial] (
        %pc = %ctrl_i32 : !fabric.bits<0> to !fabric.bits<32>,
        %pv = %publish_i32 : !fabric.bits<32>) -> !fabric.bits<32> {
      fabric.fu(
          %token = %pc : !fabric.bits<32> to !fabric.bits<0>,
          %value = %pv : !fabric.bits<32>) -> !fabric.bits<32> {
        %done, %published = fabric.op [@dataflow.sync] (%token, %value)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<32>)
              -> (!fabric.bits<0>, !fabric.bits<32>)
        fabric.yield %published : !fabric.bits<32>
      }
    }
    fabric.pe [spatial] (
        %pc = %ctrl_i8 : !fabric.bits<0> to !fabric.bits<32>,
        %pv = %publish_i8 : !fabric.bits<32>) -> !fabric.bits<32> {
      fabric.fu(
          %token = %pc : !fabric.bits<32> to !fabric.bits<0>,
          %value = %pv : !fabric.bits<32> to !fabric.bits<8>)
          -> !fabric.bits<32> {
        %done, %published = fabric.op [@dataflow.sync] (%token, %value)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<8>)
              -> (!fabric.bits<0>, !fabric.bits<8>)
        fabric.yield %published : !fabric.bits<8> to !fabric.bits<32>
      }
    }
    fabric.yield
  }
}
