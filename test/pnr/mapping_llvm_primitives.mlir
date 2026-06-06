// RUN: loom-pnr-map --dfg-mlir %s --graph zext_graph --hardware-mlir %s --hardware llvm_primitive_adg --workload zext_graph --output %t.zext.csv --artifact %t.zext.json
// RUN: loom-pnr-map --dfg-mlir %s --graph abs_graph --hardware-mlir %s --hardware llvm_primitive_adg --workload abs_graph --output %t.abs.csv --artifact %t.abs.json
// RUN: loom-pnr-map --dfg-mlir %s --graph fmuladd_graph --hardware-mlir %s --hardware llvm_primitive_adg --workload fmuladd_graph --output %t.fmuladd.csv --artifact %t.fmuladd.json
// RUN: loom-pnr-map --dfg-mlir %s --graph fshl_graph --hardware-mlir %s --hardware llvm_primitive_adg --workload fshl_graph --output %t.fshl.csv --artifact %t.fshl.json
// RUN: loom-pnr-map --dfg-mlir %s --graph bswap_graph --hardware-mlir %s --hardware llvm_primitive_adg --workload bswap_graph --output %t.bswap.csv --artifact %t.bswap.json
// RUN: FileCheck %s --check-prefix=ZEXT < %t.zext.csv
// RUN: FileCheck %s --check-prefix=ABS < %t.abs.csv
// RUN: FileCheck %s --check-prefix=FMULADD < %t.fmuladd.csv
// RUN: FileCheck %s --check-prefix=FSHL < %t.fshl.csv
// RUN: FileCheck %s --check-prefix=BSWAP < %t.bswap.csv

// ZEXT: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// ZEXT-NEXT: zext_graph,llvm_primitive_adg,zext_graph__llvm_primitive_adg,1,0,0,0,pass

// ABS: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// ABS-NEXT: abs_graph,llvm_primitive_adg,abs_graph__llvm_primitive_adg,1,0,0,0,pass

// FMULADD: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// FMULADD-NEXT: fmuladd_graph,llvm_primitive_adg,fmuladd_graph__llvm_primitive_adg,1,0,0,0,pass

// FSHL: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// FSHL-NEXT: fshl_graph,llvm_primitive_adg,fshl_graph__llvm_primitive_adg,1,0,0,0,pass

// BSWAP: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// BSWAP-NEXT: bswap_graph,llvm_primitive_adg,bswap_graph__llvm_primitive_adg,1,0,0,0,pass

module {
  dataflow.graph.func private @zext_graph(%ctrl: none, %narrow: i32)
      -> (none, i64) {
    %wide = llvm.zext %narrow : i32 to i64
    dataflow.graph.return %ctrl, %wide : none, i64
  }

  dataflow.graph.func private @abs_graph(%ctrl: none, %value: i32)
      -> (none, i32) {
    %abs = "llvm.intr.abs"(%value) <{is_int_min_poison = true}> : (i32) -> i32
    dataflow.graph.return %ctrl, %abs : none, i32
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

  fabric.module @llvm_primitive_adg(%i32a : !fabric.bits<32>,
                                    %i32b : !fabric.bits<32>,
                                    %i32c : !fabric.bits<32>) {
    fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>) -> !fabric.bits<32> {
      fabric.fu(%value = %pa : !fabric.bits<32>) -> () {
        %wide = fabric.op [@llvm.zext] (%value)
                : (!fabric.bits<32>) -> !fabric.bits<64>
        fabric.yield
      }
    }
    fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>) -> !fabric.bits<32> {
      fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
        %abs = fabric.op [@llvm.intr.abs] (%value)
               : (!fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %abs : !fabric.bits<32>
      }
    }
    fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                      %pb = %i32b : !fabric.bits<32>,
                      %pc = %i32c : !fabric.bits<32>) -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>,
                %acc = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
        %result = fabric.op [@llvm.intr.fmuladd] (%lhs, %rhs, %acc)
                  : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
                    -> !fabric.bits<32>
        fabric.yield %result : !fabric.bits<32>
      }
    }
    fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>,
                      %pb = %i32b : !fabric.bits<32>,
                      %pc = %i32c : !fabric.bits<32>) -> !fabric.bits<32> {
      fabric.fu(%lhs = %pa : !fabric.bits<32>,
                %rhs = %pb : !fabric.bits<32>,
                %amount = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
        %result = fabric.op [@llvm.intr.fshl] (%lhs, %rhs, %amount)
                  : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
                    -> !fabric.bits<32>
        fabric.yield %result : !fabric.bits<32>
      }
    }
    fabric.pe [spatial] (%pa = %i32a : !fabric.bits<32>) -> !fabric.bits<32> {
      fabric.fu(%value = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
        %result = fabric.op [@llvm.intr.bswap] (%value)
                  : (!fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %result : !fabric.bits<32>
      }
    }
    fabric.yield
  }
}
