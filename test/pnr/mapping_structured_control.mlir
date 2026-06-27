// RUN: loom-pnr-map --dfg-mlir %s --graph structured_for_map --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload structured_for_map --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: structured_for_map,shared_reduction_adg,structured_for_map__structured_for_map__shared_reduction_adg,2,1,0,0,pass

// JSON-DAG: "status": "pass"
// JSON-DAG: "operation": "dataflow.load"
// JSON-DAG: "operation": "arith.addi"
// JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.addi#0.operand1"
// JSON-NOT: "unsupported PnR graph operation: scf.for"
// JSON-NOT: ".out"
// JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph structured_extui_map --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload structured_extui_map --output %t.extui.mapping.csv --artifact %t.extui.mapping.json
// RUN: FileCheck %s --check-prefix=EXTUI-CSV < %t.extui.mapping.csv
// RUN: FileCheck %s --check-prefix=EXTUI-JSON < %t.extui.mapping.json

// EXTUI-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// EXTUI-CSV-NEXT: structured_extui_map,shared_reduction_adg,structured_extui_map__structured_extui_map__shared_reduction_adg,3,2,0,0,pass

// EXTUI-JSON-DAG: "status": "pass"
// EXTUI-JSON-DAG: "operation": "arith.cmpi"
// EXTUI-JSON-DAG: "operation": "arith.extui"
// EXTUI-JSON-DAG: "operation": "arith.addi"
// EXTUI-JSON-DAG: "edge_ref": "arith.cmpi#0.result0->arith.extui#0.operand0"
// EXTUI-JSON-DAG: "edge_ref": "arith.extui#0.result0->arith.addi#0.operand1"
// EXTUI-JSON-NOT: "unsupported PnR graph operation: arith.extui"
// EXTUI-JSON-NOT: ".out"
// EXTUI-JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph structured_index_castui_map --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload structured_index_castui_map --output %t.index-castui.mapping.csv --artifact %t.index-castui.mapping.json
// RUN: FileCheck %s --check-prefix=INDEX-CASTUI-CSV < %t.index-castui.mapping.csv
// RUN: FileCheck %s --check-prefix=INDEX-CASTUI-JSON < %t.index-castui.mapping.json

// INDEX-CASTUI-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// INDEX-CASTUI-CSV-NEXT: structured_index_castui_map,shared_reduction_adg,structured_index_castui_map__structured_index_castui_map__shared_reduction_adg,4,4,0,0,pass

// INDEX-CASTUI-JSON-DAG: "status": "pass"
// INDEX-CASTUI-JSON-DAG: "operation": "arith.cmpi"
// INDEX-CASTUI-JSON-DAG: "operation": "arith.extui"
// INDEX-CASTUI-JSON-DAG: "operation": "dataflow.load"
// INDEX-CASTUI-JSON-DAG: "operation": "arith.addi"
// INDEX-CASTUI-JSON-DAG: "edge_ref": "arith.extui#0.result0->dataflow.load#0.operand1"
// INDEX-CASTUI-JSON-NOT: "operation": "arith.index_castui"
// INDEX-CASTUI-JSON-NOT: "unsupported PnR graph operation: arith.index_castui"
// INDEX-CASTUI-JSON-NOT: ".out"
// INDEX-CASTUI-JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph structured_cmpi_ugt_map --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload structured_cmpi_ugt_map --output %t.cmpi-ugt.mapping.csv --artifact %t.cmpi-ugt.mapping.json
// RUN: FileCheck %s --check-prefix=CMPI-UGT-CSV < %t.cmpi-ugt.mapping.csv
// RUN: FileCheck %s --check-prefix=CMPI-UGT-JSON < %t.cmpi-ugt.mapping.json

// CMPI-UGT-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CMPI-UGT-CSV-NEXT: structured_cmpi_ugt_map,shared_reduction_adg,structured_cmpi_ugt_map__structured_cmpi_ugt_map__shared_reduction_adg,3,2,0,0,pass

// CMPI-UGT-JSON-DAG: "status": "pass"
// CMPI-UGT-JSON-DAG: "operation": "arith.cmpi"
// CMPI-UGT-JSON-DAG: "operation": "arith.extui"
// CMPI-UGT-JSON-DAG: "operation": "arith.addi"
// CMPI-UGT-JSON-DAG: "sw_configs.predicate"
// CMPI-UGT-JSON-DAG: "ugt"
// CMPI-UGT-JSON-NOT: "missing hardware resource for software op arith.cmpi"

// RUN: loom-pnr-map --dfg-mlir %s --graph structured_cmpi_i64_extui_map --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload structured_cmpi_i64_extui_map --output %t.cmpi-i64-extui.mapping.csv --artifact %t.cmpi-i64-extui.mapping.json
// RUN: FileCheck %s --check-prefix=CMPI-I64-EXTUI-CSV < %t.cmpi-i64-extui.mapping.csv
// RUN: FileCheck %s --check-prefix=CMPI-I64-EXTUI-JSON < %t.cmpi-i64-extui.mapping.json

// CMPI-I64-EXTUI-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CMPI-I64-EXTUI-CSV-NEXT: structured_cmpi_i64_extui_map,shared_reduction_adg,structured_cmpi_i64_extui_map__structured_cmpi_i64_extui_map__shared_reduction_adg,3,2,0,0,pass

// CMPI-I64-EXTUI-JSON-DAG: "status": "pass"
// CMPI-I64-EXTUI-JSON-DAG: "edge_ref": "arith.cmpi#0.result0->arith.extui#0.operand0"
// CMPI-I64-EXTUI-JSON-DAG: "edge_ref": "arith.extui#0.result0->llvm.trunc#0.operand0"
// CMPI-I64-EXTUI-JSON-NOT: ".out"
// CMPI-I64-EXTUI-JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph structured_cmpf_xori_extui_map --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload structured_cmpf_xori_extui_map --output %t.cmpf-xori-extui.mapping.csv --artifact %t.cmpf-xori-extui.mapping.json
// RUN: FileCheck %s --check-prefix=CMPF-XORI-CSV < %t.cmpf-xori-extui.mapping.csv
// RUN: FileCheck %s --check-prefix=CMPF-XORI-JSON < %t.cmpf-xori-extui.mapping.json

// CMPF-XORI-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CMPF-XORI-CSV-NEXT: structured_cmpf_xori_extui_map,shared_reduction_adg,structured_cmpf_xori_extui_map__structured_cmpf_xori_extui_map__shared_reduction_adg,3,2,0,0,pass

// CMPF-XORI-JSON-DAG: "status": "pass"
// CMPF-XORI-JSON-DAG: "edge_ref": "arith.cmpf#0.result0->arith.xori#0.operand0"
// CMPF-XORI-JSON-DAG: "edge_ref": "arith.xori#0.result0->arith.extui#0.operand0"

// RUN: loom-pnr-map --dfg-mlir %s --graph structured_load_store_forward_map --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload structured_load_store_forward_map --output %t.load-store-forward.mapping.csv --artifact %t.load-store-forward.mapping.json
// RUN: FileCheck %s --check-prefix=LOAD-STORE-CSV < %t.load-store-forward.mapping.csv
// RUN: FileCheck %s --check-prefix=LOAD-STORE-JSON < %t.load-store-forward.mapping.json

// LOAD-STORE-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// LOAD-STORE-CSV-NEXT: structured_load_store_forward_map,shared_reduction_adg,structured_load_store_forward_map__structured_load_store_forward_map__shared_reduction_adg,6,2,0,0,pass

// LOAD-STORE-JSON-DAG: "status": "pass"
// LOAD-STORE-JSON-DAG: "edge_ref": "dataflow.load#2.result0->dataflow.store#0.operand2"
// LOAD-STORE-JSON-DAG: "edge_ref": "dataflow.load#3.result0->dataflow.store#1.operand2"

// RUN: loom-pnr-map --dfg-mlir %s --graph structured_if_result_forward_map --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload structured_if_result_forward_map --output %t.if-result-forward.mapping.csv --artifact %t.if-result-forward.mapping.json
// RUN: FileCheck %s --check-prefix=IF-RESULT-CSV < %t.if-result-forward.mapping.csv
// RUN: FileCheck %s --check-prefix=IF-RESULT-JSON < %t.if-result-forward.mapping.json

// IF-RESULT-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// IF-RESULT-CSV-NEXT: structured_if_result_forward_map,shared_reduction_adg,structured_if_result_forward_map__structured_if_result_forward_map__shared_reduction_adg,3,2,0,0,pass

// IF-RESULT-JSON-DAG: "status": "pass"
// IF-RESULT-JSON-DAG: "edge_ref": "arith.cmpi#0.result0->arith.extui#0.operand0"
// IF-RESULT-JSON-DAG: "edge_ref": "arith.extui#0.result0->arith.addi#0.operand0"
// IF-RESULT-JSON-NOT: ".out"
// IF-RESULT-JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph structured_forall_store_map --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload structured_forall_store_map --output %t.forall-store.mapping.csv --artifact %t.forall-store.mapping.json
// RUN: FileCheck %s --check-prefix=FORALL-STORE-CSV < %t.forall-store.mapping.csv
// RUN: FileCheck %s --check-prefix=FORALL-STORE-JSON < %t.forall-store.mapping.json

// FORALL-STORE-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// FORALL-STORE-CSV-NEXT: structured_forall_store_map,shared_reduction_adg,structured_forall_store_map__structured_forall_store_map__shared_reduction_adg,3,2,0,0,pass

// FORALL-STORE-JSON-DAG: "status": "pass"
// FORALL-STORE-JSON-DAG: "operation": "dataflow.load"
// FORALL-STORE-JSON-DAG: "operation": "arith.addi"
// FORALL-STORE-JSON-DAG: "operation": "dataflow.store"
// FORALL-STORE-JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.addi#0.operand0"
// FORALL-STORE-JSON-DAG: "edge_ref": "arith.addi#0.result0->dataflow.store#0.operand2"
// FORALL-STORE-JSON-NOT: "unsupported PnR graph operation: scf.forall"
// FORALL-STORE-JSON-NOT: ".out"
// FORALL-STORE-JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph structured_while_condition_forward_map --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload structured_while_condition_forward_map --output %t.while-condition-forward.mapping.csv --artifact %t.while-condition-forward.mapping.json
// RUN: FileCheck %s --check-prefix=WHILE-CONDITION-CSV < %t.while-condition-forward.mapping.csv
// RUN: FileCheck %s --check-prefix=WHILE-CONDITION-JSON < %t.while-condition-forward.mapping.json

// WHILE-CONDITION-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// WHILE-CONDITION-CSV-NEXT: structured_while_condition_forward_map,shared_reduction_adg,structured_while_condition_forward_map__structured_while_condition_forward_map__shared_reduction_adg,4,2,0,0,pass

// WHILE-CONDITION-JSON-DAG: "status": "pass"
// WHILE-CONDITION-JSON-DAG: "edge_ref": "arith.addi#0.result0->dataflow.store#0.operand2"
// WHILE-CONDITION-JSON-NOT: "edge_ref": "arith.subi#0.result0->dataflow.store#0.operand2"
// WHILE-CONDITION-JSON-NOT: ".out"
// WHILE-CONDITION-JSON-NOT: ".in"

// RUN: not loom-pnr-map --dfg-mlir %s --graph structured_forall_shared_out_map --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload structured_forall_shared_out_map --output %t.forall-shared-out.mapping.csv --artifact %t.forall-shared-out.mapping.json 2>&1 | FileCheck %s --check-prefix=FORALL-SHARED-OUT-ERR

// FORALL-SHARED-OUT-ERR: graph contains unsupported operation for PnR mapping: scf.forall

module {
  dataflow.graph.func private @structured_for_map(%ctrl: none, %lb: i32,
      %ub: i32, %step: i32, %init: i32, %mem: memref<?xi32>) -> (none, i32) {
    %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (i32) : i32 {
      %idx = arith.index_cast %i : i32 to index
      %data, %done = dataflow.load %mem[%idx] %ctrl : memref<?xi32>
      %next = arith.addi %acc, %data : i32
      scf.yield %next : i32
    }
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  dataflow.graph.func private @structured_extui_map(%ctrl: none, %lb: i32,
      %ub: i32, %step: i32, %init: i32, %limit: i32) -> (none, i32) {
    %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (i32) : i32 {
      %under_limit = arith.cmpi ult, %i, %limit : i32
      %increment = arith.extui %under_limit : i1 to i32
      %next = arith.addi %acc, %increment : i32
      scf.yield %next : i32
    }
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  dataflow.graph.func private @structured_index_castui_map(%ctrl: none,
      %lb: i32, %ub: i32, %step: i32, %init: i32, %limit: i32,
      %mem: memref<?xi32>) -> (none, i32) {
    %sum = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (i32) : i32 {
      %under_limit = arith.cmpi ult, %i, %limit : i32
      %increment = arith.extui %under_limit : i1 to i32
      %idx = arith.index_castui %increment : i32 to index
      %data, %done = dataflow.load %mem[%idx] %ctrl : memref<?xi32>
      %biased = arith.addi %data, %increment : i32
      scf.yield %biased : i32
    }
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  dataflow.graph.func private @structured_cmpi_ugt_map(%ctrl: none, %lhs: i32,
      %rhs: i32, %bias: i32) -> (none, i32) {
    %pred = arith.cmpi ugt, %lhs, %rhs : i32
    %as_i32 = arith.extui %pred : i1 to i32
    %sum = arith.addi %bias, %as_i32 : i32
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  dataflow.graph.func private @structured_cmpi_i64_extui_map(%ctrl: none,
      %lhs: i64, %rhs: i64) -> (none, i32) {
    %pred = arith.cmpi ult, %lhs, %rhs : i64
    %wide = arith.extui %pred : i1 to i64
    %narrow = llvm.trunc %wide : i64 to i32
    dataflow.graph.return %ctrl, %narrow : none, i32
  }

  dataflow.graph.func private @structured_cmpf_xori_extui_map(%ctrl: none,
      %lhs: f32, %rhs: f32, %flag: i1) -> (none, i32) {
    %pred = arith.cmpf ugt, %lhs, %rhs : f32
    %masked = arith.xori %pred, %flag : i1
    %as_i32 = arith.extui %masked : i1 to i32
    dataflow.graph.return %ctrl, %as_i32 : none, i32
  }

  dataflow.graph.func private @structured_load_store_forward_map(%ctrl: none,
      %mem: memref<?xi32>, %dst: memref<?xi32>, %i0: index, %i1: index,
      %i2: index, %i3: index) -> none {
    %d0, %done0 = dataflow.load %mem[%i0] %ctrl : memref<?xi32>
    %d1, %done1 = dataflow.load %mem[%i1] %ctrl : memref<?xi32>
    %d2, %done2 = dataflow.load %mem[%i2] %ctrl : memref<?xi32>
    %d3, %done3 = dataflow.load %mem[%i3] %ctrl : memref<?xi32>
    %store0 = dataflow.store %dst[%i2] %d2 %ctrl : memref<?xi32>
    %store1 = dataflow.store %dst[%i3] %d3 %ctrl : memref<?xi32>
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @structured_if_result_forward_map(%ctrl: none,
      %cond: i1, %a: i32, %b: i32, %bias: i32, %tail: i32) -> (none, i32) {
    %branch = scf.if %cond -> (i32) {
      %pred = arith.cmpi ult, %a, %b : i32
      %then = arith.extui %pred : i1 to i32
      scf.yield %then : i32
    } else {
      scf.yield %bias : i32
    }
    %out = arith.addi %branch, %tail : i32
    dataflow.graph.return %ctrl, %out : none, i32
  }

  dataflow.graph.func private @structured_forall_store_map(%ctrl: none,
      %lb: index, %ub: index, %mem: memref<?xi32>, %addend: i32) -> none {
    scf.forall (%i) = (%lb) to (%ub) step (1) {
      %value, %done = dataflow.load %mem[%i] %ctrl : memref<?xi32>
      %stored = arith.addi %value, %addend : i32
      %store_done = dataflow.store %mem[%i] %stored %ctrl : memref<?xi32>
    }
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @structured_while_condition_forward_map(
      %ctrl: none, %iv0: i32, %ub: i32, %step: i32, %out: memref<?xi32>)
      -> none {
    %result = scf.while (%iv = %iv0) : (i32) -> i32 {
      %next = arith.addi %iv, %step : i32
      %cont = arith.cmpi slt, %next, %ub : i32
      scf.condition(%cont) %next : i32
    } do {
    ^bb0(%carried: i32):
      %decoy = arith.subi %carried, %step : i32
      scf.yield %decoy : i32
    }
    %idx = arith.index_cast %iv0 : i32 to index
    %done = dataflow.store %out[%idx] %result %ctrl : memref<?xi32>
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @structured_forall_shared_out_map(%ctrl: none,
      %init: tensor<2xi32>) -> (none, tensor<2xi32>) {
    %result = scf.forall (%i) in (2) shared_outs(%out = %init)
        -> (tensor<2xi32>) {
      scf.forall.in_parallel {
      }
    }
    dataflow.graph.return %ctrl, %result : none, tensor<2xi32>
  }
}
