// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph arm_qsub16 --hardware-mlir %s --hardware arm_intrinsic_adg --workload arm_qsub16 --output %t.csv --artifact %t.json
// RUN: FileCheck %s --check-prefix=CSV < %t.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.json
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph arm_qsub8 --hardware-mlir %s --hardware arm_intrinsic_adg --workload arm_qsub8 --output %t.qsub8.csv --artifact %t.qsub8.json
// RUN: FileCheck %s --check-prefix=QSUB8-CSV < %t.qsub8.csv
// RUN: FileCheck %s --check-prefix=QSUB8-JSON < %t.qsub8.json
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph arm_qadd16 --hardware-mlir %s --hardware arm_intrinsic_adg --workload arm_qadd16 --output %t.qadd16.csv --artifact %t.qadd16.json
// RUN: FileCheck %s --check-prefix=QADD16-CSV < %t.qadd16.csv
// RUN: FileCheck %s --check-prefix=QADD16-JSON < %t.qadd16.json
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph arm_sadd16 --hardware-mlir %s --hardware arm_intrinsic_adg --workload arm_sadd16 --output %t.sadd16.csv --artifact %t.sadd16.json
// RUN: FileCheck %s --check-prefix=SADD16-CSV < %t.sadd16.csv
// RUN: FileCheck %s --check-prefix=SADD16-JSON < %t.sadd16.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: arm_qsub16,arm_intrinsic_adg,arm_qsub16__arm_qsub16__arm_intrinsic_adg,2,1,0,0,pass

// JSON-DAG: "operation": "llvm.arm.qsub16"
// JSON-DAG: "hardware": "arm_intrinsic_adg::fabric.op#0"

// QSUB8-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// QSUB8-CSV-NEXT: arm_qsub8,arm_intrinsic_adg,arm_qsub8__arm_qsub8__arm_intrinsic_adg,2,1,0,0,pass

// QSUB8-JSON-DAG: "operation": "llvm.arm.qsub8"
// QSUB8-JSON-DAG: "hardware": "arm_intrinsic_adg::fabric.op#1"

// QADD16-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// QADD16-CSV-NEXT: arm_qadd16,arm_intrinsic_adg,arm_qadd16__arm_qadd16__arm_intrinsic_adg,2,1,0,0,pass

// QADD16-JSON-DAG: "operation": "llvm.arm.qadd16"
// QADD16-JSON-DAG: "hardware": "arm_intrinsic_adg::fabric.op#2"

// SADD16-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// SADD16-CSV-NEXT: arm_sadd16,arm_intrinsic_adg,arm_sadd16__arm_sadd16__arm_intrinsic_adg,2,1,0,0,pass

// SADD16-JSON-DAG: "operation": "llvm.arm.sadd16"
// SADD16-JSON-DAG: "hardware": "arm_intrinsic_adg::fabric.op#3"

module {
  dataflow.graph.func private @arm_qsub16(%ctrl: none, %zero: i32, %value: i32)
      -> (none, i32) {
    %packed = llvm.call_intrinsic "llvm.arm.qsub16"(%zero, %value)
        : (i32, i32) -> i32
    dataflow.graph.return %ctrl, %packed : none, i32
  }

  dataflow.graph.func private @arm_qsub8(%ctrl: none, %zero: i32, %value: i32)
      -> (none, i32) {
    %packed = llvm.call_intrinsic "llvm.arm.qsub8"(%zero, %value)
        : (i32, i32) -> i32
    dataflow.graph.return %ctrl, %packed : none, i32
  }

  dataflow.graph.func private @arm_qadd16(%ctrl: none, %lhs: i32, %rhs: i32)
      -> (none, i32) {
    %packed = llvm.call_intrinsic "llvm.arm.qadd16"(%lhs, %rhs)
        : (i32, i32) -> i32
    dataflow.graph.return %ctrl, %packed : none, i32
  }

  dataflow.graph.func private @arm_sadd16(%ctrl: none, %lhs: i32, %rhs: i32)
      -> (none, i32) {
    %packed = llvm.call_intrinsic "llvm.arm.sadd16"(%lhs, %rhs)
        : (i32, i32) -> i32
    dataflow.graph.return %ctrl, %packed : none, i32
  }

  fabric.module @arm_intrinsic_adg(%ctrl : !fabric.bits<0>,
                                   %i32a : !fabric.bits<32>,
                                   %i32b : !fabric.bits<32>) {
    %a_qsub16, %a_qsub8, %a_qadd16, %a_sadd16 =
        fabric.switch [spatial] %i32a
          [{connectivity_table = ["1", "1", "1", "1"]}]
          : (!fabric.bits<32>)
            -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                !fabric.bits<32>)
    %b_qsub16, %b_qsub8, %b_qadd16, %b_sadd16 =
        fabric.switch [spatial] %i32b
          [{connectivity_table = ["1", "1", "1", "1"]}]
          : (!fabric.bits<32>)
            -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
                !fabric.bits<32>)
    %qsub16 = fabric.pe [spatial] (%pa = %a_qsub16 : !fabric.bits<32>,
                         %pb = %b_qsub16 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%zero = %pa : !fabric.bits<32>,
                %value = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
        %result = fabric.op [@llvm.arm.qsub16] (%zero, %value)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %result : !fabric.bits<32>
      }
    }
    %qsub8 = fabric.pe [spatial] (%pa = %a_qsub8 : !fabric.bits<32>,
                         %pb = %b_qsub8 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%zero = %pa : !fabric.bits<32>,
                %value = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
        %result = fabric.op [@llvm.arm.qsub8] (%zero, %value)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %result : !fabric.bits<32>
      }
    }
    %qadd16 = fabric.pe [spatial] (%pa = %a_qadd16 : !fabric.bits<32>,
                         %pb = %b_qadd16 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
        %result = fabric.op [@llvm.arm.qadd16] (%lhs, %rhs)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %result : !fabric.bits<32>
      }
    }
    %sadd16 = fabric.pe [spatial] (%pa = %a_sadd16 : !fabric.bits<32>,
                         %pb = %b_sadd16 : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
        %result = fabric.op [@llvm.arm.sadd16] (%lhs, %rhs)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %result : !fabric.bits<32>
      }
    }
    %published_value = fabric.switch [spatial]
        %qsub16, %qsub8, %qadd16, %sadd16
        [{connectivity_table = ["1111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
           !fabric.bits<32>) -> !fabric.bits<32>
    fabric.pe [spatial] (
        %pc = %ctrl : !fabric.bits<0> to !fabric.bits<32>,
        %pv = %published_value : !fabric.bits<32>) -> !fabric.bits<32> {
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
    fabric.yield
  }
}
