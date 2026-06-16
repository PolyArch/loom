// RUN: loom-pnr-map --dfg-mlir %s --graph arm_qsub16 --hardware-mlir %s --hardware arm_intrinsic_adg --workload arm_qsub16 --output %t.csv --artifact %t.json
// RUN: FileCheck %s --check-prefix=CSV < %t.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.json
// RUN: loom-pnr-map --dfg-mlir %s --graph arm_qsub8 --hardware-mlir %s --hardware arm_intrinsic_adg --workload arm_qsub8 --output %t.qsub8.csv --artifact %t.qsub8.json
// RUN: FileCheck %s --check-prefix=QSUB8-CSV < %t.qsub8.csv
// RUN: FileCheck %s --check-prefix=QSUB8-JSON < %t.qsub8.json
// RUN: loom-pnr-map --dfg-mlir %s --graph arm_qadd16 --hardware-mlir %s --hardware arm_intrinsic_adg --workload arm_qadd16 --output %t.qadd16.csv --artifact %t.qadd16.json
// RUN: FileCheck %s --check-prefix=QADD16-CSV < %t.qadd16.csv
// RUN: FileCheck %s --check-prefix=QADD16-JSON < %t.qadd16.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: arm_qsub16,arm_intrinsic_adg,arm_qsub16__arm_qsub16__arm_intrinsic_adg,1,0,0,0,pass

// JSON-DAG: "operation": "llvm.arm.qsub16"
// JSON-DAG: "hardware": "arm_intrinsic_adg::fabric.op#0"

// QSUB8-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// QSUB8-CSV-NEXT: arm_qsub8,arm_intrinsic_adg,arm_qsub8__arm_qsub8__arm_intrinsic_adg,1,0,0,0,pass

// QSUB8-JSON-DAG: "operation": "llvm.arm.qsub8"
// QSUB8-JSON-DAG: "hardware": "arm_intrinsic_adg::fabric.op#1"

// QADD16-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// QADD16-CSV-NEXT: arm_qadd16,arm_intrinsic_adg,arm_qadd16__arm_qadd16__arm_intrinsic_adg,1,0,0,0,pass

// QADD16-JSON-DAG: "operation": "llvm.arm.qadd16"
// QADD16-JSON-DAG: "hardware": "arm_intrinsic_adg::fabric.op#2"

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

  fabric.module @arm_intrinsic_adg(%i32a : !fabric.bits<32>,
                                   %i32b : !fabric.bits<32>) {
    fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                         %pb = %i32b : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%zero = %pa : !fabric.bits<32>,
                %value = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
        %result = fabric.op [@llvm.arm.qsub16] (%zero, %value)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %result : !fabric.bits<32>
      }
    }
    fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                         %pb = %i32b : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%zero = %pa : !fabric.bits<32>,
                %value = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
        %result = fabric.op [@llvm.arm.qsub8] (%zero, %value)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %result : !fabric.bits<32>
      }
    }
    fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                         %pb = %i32b : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
        %result = fabric.op [@llvm.arm.qadd16] (%lhs, %rhs)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %result : !fabric.bits<32>
      }
    }
    fabric.yield
  }
}
