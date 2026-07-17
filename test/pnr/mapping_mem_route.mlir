// RUN: loom-pnr-map --dfg-mlir %s --graph mem_route --hardware-mlir %s --hardware mem_route_adg --workload mem_route --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: mem_route,mem_route_adg,mem_route__mem_route__mem_route_adg,3,2,0,0,pass

// JSON-DAG: "status": "pass"
// JSON-DAG: "hardware": "mem_route_adg::mem.load#0"
// JSON-DAG: "segment_kind": "resource_edge"
// JSON-DAG: "source_endpoint": "mem_route_adg::mem.load#0.result0"
// JSON-DAG: "sink_endpoint": "mem_route_adg::fabric.op#0.operand0"
// JSON-DAG: "source_endpoint": "mem_route_adg::mem.load#0.result1"
// JSON-DAG: "sink_endpoint": "mem_route_adg::fabric.op#1.operand0"
// JSON-NOT: ".out"
// JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph mem_route --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mem_route_shared_sync_prefix --output %t.shared-sync.mapping.csv --artifact %t.shared-sync.mapping.json
// RUN: FileCheck %s --check-prefix=SHARED-SYNC-CSV < %t.shared-sync.mapping.csv
// RUN: FileCheck %s --check-prefix=SHARED-SYNC-JSON < %t.shared-sync.mapping.json

// SHARED-SYNC-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// SHARED-SYNC-CSV-NEXT: mem_route_shared_sync_prefix,shared_reduction_adg,mem_route_shared_sync_prefix__mem_route__shared_reduction_adg,3,2,0,0,pass

// SHARED-SYNC-JSON-DAG: "status": "pass"
// SHARED-SYNC-JSON-DAG: "operation": "dataflow.sync"
// SHARED-SYNC-JSON-DAG: "edge_ref": "dataflow.load#0.result1->dataflow.sync#0.operand0"
// SHARED-SYNC-JSON-NOT: ".out"
// SHARED-SYNC-JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph mem_two_loads_one_port --hardware-mlir %s --hardware mem_route_adg --workload mem_two_loads_one_port --output %t.twoload.mapping.csv --artifact %t.twoload.mapping.json
// RUN: FileCheck %s --check-prefix=TWOLOAD-CSV < %t.twoload.mapping.csv
// RUN: FileCheck %s --check-prefix=TWOLOAD-JSON < %t.twoload.mapping.json

// TWOLOAD-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// TWOLOAD-CSV-NEXT: mem_two_loads_one_port,mem_route_adg,mem_two_loads_one_port__mem_two_loads_one_port__mem_route_adg,2,1,0,1,fail,missing hardware resource for software op dataflow.load

// TWOLOAD-JSON-DAG: "status": "fail"
// TWOLOAD-JSON-DAG: missing hardware resource for software op dataflow.load
// TWOLOAD-JSON-DAG: "operation": "dataflow.load"
// TWOLOAD-JSON-DAG: "resource_kind": "fabric.mem.load"
// TWOLOAD-JSON-DAG: "unplaced_records": 1
// TWOLOAD-JSON-NOT: "fabric.mem.copy"
// TWOLOAD-JSON-NOT: "memory_copy_binding"

// RUN: loom-pnr-map --dfg-mlir %s --graph mem_store_route --hardware-mlir %s --hardware mem_store_route_adg --workload mem_store_route --output %t.store.mapping.csv --artifact %t.store.mapping.json
// RUN: FileCheck %s --check-prefix=STORE-CSV < %t.store.mapping.csv
// RUN: FileCheck %s --check-prefix=STORE-JSON < %t.store.mapping.json

// STORE-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// STORE-CSV-NEXT: mem_store_route,mem_store_route_adg,mem_store_route__mem_store_route__mem_store_route_adg,2,2,0,0,pass

// STORE-JSON-DAG: "status": "pass"
// STORE-JSON-DAG: "hardware": "mem_store_route_adg::mem.load#0"
// STORE-JSON-DAG: "hardware": "mem_store_route_adg::mem.store#0"
// STORE-JSON-DAG: "source_endpoint": "mem_store_route_adg::mem.load#0.result0"
// STORE-JSON-DAG: "sink_endpoint": "mem_store_route_adg::mem.store#0.operand1"
// STORE-JSON-DAG: "source_endpoint": "mem_store_route_adg::mem.load#0.result1"
// STORE-JSON-DAG: "sink_endpoint": "mem_store_route_adg::mem.store#0.operand2"
// STORE-JSON-NOT: "sink_endpoint": "mem_store_route_adg::mem.store#0.operand3"
// STORE-JSON-NOT: ".out"
// STORE-JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph mem_gep_store --hardware-mlir %s --hardware mem_store_route_adg --workload mem_gep_store --output %t.gep.mapping.csv --artifact %t.gep.mapping.json
// RUN: FileCheck %s --check-prefix=GEP-CSV < %t.gep.mapping.csv
// RUN: FileCheck %s --check-prefix=GEP-JSON < %t.gep.mapping.json

// GEP-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// GEP-CSV-NEXT: mem_gep_store,mem_store_route_adg,mem_gep_store__mem_gep_store__mem_store_route_adg,2,2,0,0,pass

// GEP-JSON-DAG: "status": "pass"
// GEP-JSON-DAG: "edge_ref": "dataflow.load#0.result0->dataflow.store#0.operand2"
// GEP-JSON-DAG: "edge_ref": "dataflow.load#0.result1->dataflow.store#0.operand3"
// GEP-JSON-NOT: "operation": "llvm.getelementptr"
// GEP-JSON-NOT: ".out"
// GEP-JSON-NOT: ".in"

// RUN: not loom-pnr-map --dfg-mlir %s --graph mem_gep_pointer_store_value --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mem_gep_pointer_store_value --output %t.gep-store-value.mapping.csv --artifact %t.gep-store-value.mapping.json 2>&1 | FileCheck %s --check-prefix=GEPSTOREVAL

// GEPSTOREVAL: graph contains unsupported operation for PnR mapping: llvm.getelementptr

// RUN: loom-pnr-map --dfg-mlir %s --graph mem_scf_pointer_yield_bookkeeping --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mem_scf_pointer_yield_bookkeeping --output %t.scfgep.mapping.csv --artifact %t.scfgep.mapping.json
// RUN: FileCheck %s --check-prefix=SCFGEP-CSV < %t.scfgep.mapping.csv
// RUN: FileCheck %s --check-prefix=SCFGEP-JSON < %t.scfgep.mapping.json

// SCFGEP-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// SCFGEP-CSV-NEXT: mem_scf_pointer_yield_bookkeeping,shared_reduction_adg,mem_scf_pointer_yield_bookkeeping__mem_scf_pointer_yield_bookkeeping__shared_reduction_adg,2,2,0,0,pass

// SCFGEP-JSON-DAG: "status": "pass"
// SCFGEP-JSON-DAG: "edge_ref": "dataflow.load#0.result0->dataflow.store#0.operand2"
// SCFGEP-JSON-DAG: "edge_ref": "dataflow.load#0.result1->dataflow.store#0.operand3"
// SCFGEP-JSON-NOT: "operation": "llvm.getelementptr"
// SCFGEP-JSON-NOT: ".out"
// SCFGEP-JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph mem_nested_scf_pointer_yield_bookkeeping --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mem_nested_scf_pointer_yield_bookkeeping --output %t.nested-scfgep.mapping.csv --artifact %t.nested-scfgep.mapping.json
// RUN: FileCheck %s --check-prefix=NESTED-SCFGEP-CSV < %t.nested-scfgep.mapping.csv
// RUN: FileCheck %s --check-prefix=NESTED-SCFGEP-JSON < %t.nested-scfgep.mapping.json

// NESTED-SCFGEP-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// NESTED-SCFGEP-CSV-NEXT: mem_nested_scf_pointer_yield_bookkeeping,shared_reduction_adg,mem_nested_scf_pointer_yield_bookkeeping__mem_nested_scf_pointer_yield_bookkeeping__shared_reduction_adg,{{[1-9][0-9]*}},{{[1-9][0-9]*}},0,0,pass

// NESTED-SCFGEP-JSON-DAG: "status": "pass"
// NESTED-SCFGEP-JSON-DAG: "operation": "dataflow.load"
// NESTED-SCFGEP-JSON-DAG: "operation": "dataflow.store"
// NESTED-SCFGEP-JSON-NOT: "operation": "llvm.getelementptr"
// NESTED-SCFGEP-JSON-NOT: ".out"
// NESTED-SCFGEP-JSON-NOT: ".in"

// RUN: not loom-pnr-map --dfg-mlir %s --graph mem_nested_scf_pointer_store_value --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mem_nested_scf_pointer_store_value --output %t.nested-store-value.mapping.csv --artifact %t.nested-store-value.mapping.json 2>&1 | FileCheck %s --check-prefix=NESTEDSTOREVAL

// NESTEDSTOREVAL: graph contains unsupported operation for PnR mapping: llvm.getelementptr

// RUN: not loom-pnr-map --dfg-mlir %s --graph mem_scf_pointer_semantic_return --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mem_scf_pointer_semantic_return --output %t.scfptrret.mapping.csv --artifact %t.scfptrret.mapping.json 2>&1 | FileCheck %s --check-prefix=SCFPTRRET

// SCFPTRRET: graph returns unsupported pointer value for PnR mapping

// RUN: loom-pnr-map --dfg-mlir %s --graph mem_pointer_bookkeeping --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mem_pointer_bookkeeping --output %t.ptr.mapping.csv --artifact %t.ptr.mapping.json
// RUN: FileCheck %s --check-prefix=PTR-CSV < %t.ptr.mapping.csv
// RUN: FileCheck %s --check-prefix=PTR-JSON < %t.ptr.mapping.json

// PTR-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// PTR-CSV-NEXT: mem_pointer_bookkeeping,shared_reduction_adg,mem_pointer_bookkeeping__mem_pointer_bookkeeping__shared_reduction_adg,5,6,0,0,pass

// PTR-JSON-DAG: "status": "pass"
// PTR-JSON-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.load#0.operand1"
// PTR-JSON-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.store#0.operand1"
// PTR-JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.addf#0.operand0"
// PTR-JSON-DAG: "edge_ref": "arith.addf#0.result0->dataflow.store#0.operand2"
// PTR-JSON-DAG: "edge_ref": "dataflow.store#0.result0->dataflow.sync#0.operand1"
// PTR-JSON-NOT: "operation": "llvm.getelementptr"
// PTR-JSON-NOT: "operation": "dataflow.carry"
// PTR-JSON-NOT: ".out"
// PTR-JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph mem_pointer_bookkeeping_return --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mem_pointer_bookkeeping_return --output %t.ptrbookret.mapping.csv --artifact %t.ptrbookret.mapping.json
// RUN: FileCheck %s --check-prefix=PTRBOOKRET-CSV < %t.ptrbookret.mapping.csv
// RUN: FileCheck %s --check-prefix=PTRBOOKRET-JSON < %t.ptrbookret.mapping.json

// PTRBOOKRET-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// PTRBOOKRET-CSV-NEXT: mem_pointer_bookkeeping_return,shared_reduction_adg,mem_pointer_bookkeeping_return__mem_pointer_bookkeeping_return__shared_reduction_adg,5,6,0,0,pass

// PTRBOOKRET-JSON-DAG: "status": "pass"
// PTRBOOKRET-JSON-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.load#0.operand1"
// PTRBOOKRET-JSON-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.store#0.operand1"
// PTRBOOKRET-JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.addf#0.operand0"
// PTRBOOKRET-JSON-DAG: "edge_ref": "arith.addf#0.result0->dataflow.store#0.operand2"
// PTRBOOKRET-JSON-DAG: "edge_ref": "dataflow.store#0.result0->dataflow.sync#0.operand1"
// PTRBOOKRET-JSON-NOT: "operation": "llvm.getelementptr"
// PTRBOOKRET-JSON-NOT: "operation": "dataflow.carry"
// PTRBOOKRET-JSON-NOT: ".out"
// PTRBOOKRET-JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph mem_gep_bookkeeping_return --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mem_gep_bookkeeping_return --output %t.gepbookret.mapping.csv --artifact %t.gepbookret.mapping.json
// RUN: FileCheck %s --check-prefix=GEPBOOKRET-CSV < %t.gepbookret.mapping.csv
// RUN: FileCheck %s --check-prefix=GEPBOOKRET-JSON < %t.gepbookret.mapping.json

// GEPBOOKRET-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// GEPBOOKRET-CSV-NEXT: mem_gep_bookkeeping_return,shared_reduction_adg,mem_gep_bookkeeping_return__mem_gep_bookkeeping_return__shared_reduction_adg,5,6,0,0,pass

// GEPBOOKRET-JSON-DAG: "status": "pass"
// GEPBOOKRET-JSON-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.load#0.operand1"
// GEPBOOKRET-JSON-DAG: "edge_ref": "dataflow.stream#0.result0->dataflow.store#0.operand1"
// GEPBOOKRET-JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.addf#0.operand0"
// GEPBOOKRET-JSON-DAG: "edge_ref": "arith.addf#0.result0->dataflow.store#0.operand2"
// GEPBOOKRET-JSON-DAG: "edge_ref": "dataflow.store#0.result0->dataflow.sync#0.operand1"
// GEPBOOKRET-JSON-NOT: "operation": "llvm.getelementptr"
// GEPBOOKRET-JSON-NOT: "operation": "dataflow.carry"
// GEPBOOKRET-JSON-NOT: ".out"
// GEPBOOKRET-JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph control_mux_needs_control_resource --hardware-mlir %s --hardware data_mux_only_adg --workload control_mux_type_guard --output %t.ctrlmux.mapping.csv --artifact %t.ctrlmux.mapping.json
// RUN: FileCheck %s --check-prefix=CTRLMUX-CSV < %t.ctrlmux.mapping.csv
// RUN: FileCheck %s --check-prefix=CTRLMUX-JSON < %t.ctrlmux.mapping.json

// CTRLMUX-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CTRLMUX-CSV-NEXT: control_mux_type_guard,data_mux_only_adg,control_mux_type_guard__control_mux_needs_control_resource__data_mux_only_adg,0,0,0,1,fail,missing hardware resource for software op dataflow.mux

// CTRLMUX-JSON-DAG: "status": "fail"
// CTRLMUX-JSON-DAG: missing hardware resource for software op dataflow.mux
// CTRLMUX-JSON-DAG: "unplaced_records": 1
// CTRLMUX-JSON-DAG: "placements": []
// CTRLMUX-JSON-NOT: "hardware": "data_mux_only_adg::fabric.op#0"

// RUN: loom-pnr-map --dfg-mlir %s --graph predicate_and_maps_to_transport_andi --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload predicate_and --output %t.predand.mapping.csv --artifact %t.predand.mapping.json
// RUN: FileCheck %s --check-prefix=PREDAND-JSON < %t.predand.mapping.json

// PREDAND-JSON-DAG: "status": "pass"
// PREDAND-JSON-DAG: "operation": "arith.andi"
// PREDAND-JSON-DAG: "edge_ref": "arith.cmpi#0.result0->arith.andi#0.operand0"
// PREDAND-JSON-DAG: "edge_ref": "arith.cmpi#1.result0->arith.andi#0.operand1"
// PREDAND-JSON-DAG: "edge_ref": "arith.andi#0.result0->arith.select#0.operand0"
// PREDAND-JSON-NOT: "missing hardware resource for software op arith.andi"

// RUN: loom-pnr-map --dfg-mlir %s --graph llvm_load_pointer --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload llvm_load_pointer --output %t.llvmload.mapping.csv --artifact %t.llvmload.mapping.json
// RUN: FileCheck %s --check-prefix=LLVMLOAD-CSV < %t.llvmload.mapping.csv
// RUN: FileCheck %s --check-prefix=LLVMLOAD-JSON < %t.llvmload.mapping.json

// LLVMLOAD-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// LLVMLOAD-CSV-NEXT: llvm_load_pointer,shared_reduction_adg,llvm_load_pointer__llvm_load_pointer__shared_reduction_adg,2,1,0,0,pass

// LLVMLOAD-JSON-DAG: "operation": "llvm.load"
// LLVMLOAD-JSON-DAG: "resource_kind": "fabric.mem.load"
// LLVMLOAD-JSON-DAG: "edge_ref": "llvm.load#0.result0->arith.addi#0.operand0"
// LLVMLOAD-JSON-NOT: "operation": "llvm.getelementptr"
// LLVMLOAD-JSON-NOT: ".out"
// LLVMLOAD-JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph llvm_select_pointer_map --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload llvm_select_pointer_map --output %t.llvmselect.mapping.csv --artifact %t.llvmselect.mapping.json
// RUN: FileCheck %s --check-prefix=LLVMSELECT-CSV < %t.llvmselect.mapping.csv
// RUN: FileCheck %s --check-prefix=LLVMSELECT-JSON < %t.llvmselect.mapping.json

// LLVMSELECT-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// LLVMSELECT-CSV-NEXT: llvm_select_pointer_map,shared_reduction_adg,llvm_select_pointer_map__llvm_select_pointer_map__shared_reduction_adg,4,2,0,0,pass

// LLVMSELECT-JSON-DAG: "status": "pass"
// LLVMSELECT-JSON-DAG: "operation": "llvm.select"
// LLVMSELECT-JSON-DAG: "resource_kind": "fabric.op"
// LLVMSELECT-JSON-DAG: "edge_ref": "arith.cmpi#0.result0->llvm.select#0.operand0"
// LLVMSELECT-JSON-DAG: "edge_ref": "llvm.load#0.result0->arith.addi#0.operand0"
// LLVMSELECT-JSON-NOT: "unsupported PnR graph operation: llvm.select"
// LLVMSELECT-JSON-NOT: ".out"
// LLVMSELECT-JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph llvm_select_pointer_wide_cmp_map --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload llvm_select_pointer_wide_cmp_map --output %t.llvmselectwide.mapping.csv --artifact %t.llvmselectwide.mapping.json
// RUN: FileCheck %s --check-prefix=LLVMSELECTWIDE-CSV < %t.llvmselectwide.mapping.csv
// RUN: FileCheck %s --check-prefix=LLVMSELECTWIDE-JSON < %t.llvmselectwide.mapping.json

// LLVMSELECTWIDE-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// LLVMSELECTWIDE-CSV-NEXT: llvm_select_pointer_wide_cmp_map,shared_reduction_adg,llvm_select_pointer_wide_cmp_map__llvm_select_pointer_wide_cmp_map__shared_reduction_adg,5,2,0,0,pass

// LLVMSELECTWIDE-JSON-DAG: "status": "pass"
// LLVMSELECTWIDE-JSON-DAG: "operation": "llvm.select"
// LLVMSELECTWIDE-JSON-DAG: "operation": "arith.cmpi"
// LLVMSELECTWIDE-JSON-DAG: "hardware": "shared_reduction_adg::fabric.op#77"
// LLVMSELECTWIDE-JSON-DAG: "hardware": "shared_reduction_adg::fabric.op#78"
// LLVMSELECTWIDE-JSON-DAG: "edge_ref": "arith.cmpi#1.result0->llvm.select#0.operand0"
// LLVMSELECTWIDE-JSON-DAG: "edge_ref": "llvm.load#0.result0->arith.addf#0.operand0"
// LLVMSELECTWIDE-JSON-NOT: "missing hardware resource for software op arith.cmpi"
// LLVMSELECTWIDE-JSON-NOT: ".out"
// LLVMSELECTWIDE-JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph llvm_store_pointer --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload llvm_store_pointer --output %t.llvmstore.mapping.csv --artifact %t.llvmstore.mapping.json
// RUN: FileCheck %s --check-prefix=LLVMSTORE-CSV < %t.llvmstore.mapping.csv
// RUN: FileCheck %s --check-prefix=LLVMSTORE-JSON < %t.llvmstore.mapping.json

// LLVMSTORE-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// LLVMSTORE-CSV-NEXT: llvm_store_pointer,shared_reduction_adg,llvm_store_pointer__llvm_store_pointer__shared_reduction_adg,3,2,0,0,pass

// LLVMSTORE-JSON-DAG: "operation": "llvm.fneg"
// LLVMSTORE-JSON-DAG: "resource_kind": "fabric.op"
// LLVMSTORE-JSON-DAG: "operation": "llvm.store"
// LLVMSTORE-JSON-DAG: "resource_kind": "fabric.mem.store"
// LLVMSTORE-JSON-DAG: "edge_ref": "llvm.load#0.result0->llvm.fneg#0.operand0"
// LLVMSTORE-JSON-DAG: "edge_ref": "llvm.fneg#0.result0->llvm.store#0.operand0"
// LLVMSTORE-JSON-DAG: "sink_endpoint": "shared_reduction_adg::mem.store#0.operand1"
// LLVMSTORE-JSON-NOT: "fabric.mem.copy"
// LLVMSTORE-JSON-NOT: "memory_copy_binding"
// LLVMSTORE-JSON-NOT: ".out"
// LLVMSTORE-JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph constant_addr_load_store --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload constant_addr_load_store --output %t.constload.mapping.csv --artifact %t.constload.mapping.json
// RUN: FileCheck %s --check-prefix=CONSTLOAD-CSV < %t.constload.mapping.csv
// RUN: FileCheck %s --check-prefix=CONSTLOAD-JSON < %t.constload.mapping.json

// CONSTLOAD-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CONSTLOAD-CSV-NEXT: constant_addr_load_store,shared_reduction_adg,constant_addr_load_store__constant_addr_load_store__shared_reduction_adg,5,6,0,0,pass

// CONSTLOAD-JSON-DAG: "edge_ref": "dataflow.constant#0.result0->dataflow.load#0.operand1"
// CONSTLOAD-JSON-DAG: "sink_endpoint": "shared_reduction_adg::mem.load#0.operand0"
// CONSTLOAD-JSON-DAG: "operation": "llvm.fneg"
// CONSTLOAD-JSON-DAG: "operation": "dataflow.store"
// CONSTLOAD-JSON-NOT: "fabric.mem.copy"
// CONSTLOAD-JSON-NOT: "memory_copy_binding"
// CONSTLOAD-JSON-NOT: ".out"
// CONSTLOAD-JSON-NOT: ".in"

// RUN: loom-adg-builder-test --shared-memory-reduction --output %t.cfftred3.hardware.mlir
// RUN: loom-pnr-map --dfg-mlir %s --graph cfft_red3_fmul_pair --hardware-mlir %t.cfftred3.hardware.mlir --hardware shared_memory_reduction_adg --workload cfft_red3_fmul_pair --output %t.cfftred3.mapping.csv --artifact %t.cfftred3.mapping.json
// RUN: FileCheck %s --check-prefix=CFFT-RED3-CSV < %t.cfftred3.mapping.csv
// RUN: FileCheck %s --check-prefix=CFFT-RED3-JSON < %t.cfftred3.mapping.json

// CFFT-RED3-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CFFT-RED3-CSV-NEXT: cfft_red3_fmul_pair,shared_memory_reduction_adg,cfft_red3_fmul_pair__cfft_red3_fmul_pair__shared_memory_reduction_adg,23,42,0,0,pass,mapped software graph to fabric resources

// CFFT-RED3-JSON-DAG: "status": "pass"
// CFFT-RED3-JSON-DAG: "operation": "arith.mulf"
// CFFT-RED3-JSON-DAG: "edge_ref": "dataflow.gate#1.result1->arith.mulf#0.operand0"
// CFFT-RED3-JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.mulf#0.operand1"
// CFFT-RED3-JSON-DAG: "edge_ref": "dataflow.gate#1.result1->arith.mulf#1.operand0"
// CFFT-RED3-JSON-DAG: "edge_ref": "llvm.fneg#0.result0->arith.mulf#1.operand1"
// CFFT-RED3-JSON-NOT: "unrouted"
// CFFT-RED3-JSON-NOT: ".out"
// CFFT-RED3-JSON-NOT: ".in"

// RUN: not loom-pnr-map --dfg-mlir %s --graph mem_pointer_semantic_return --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mem_pointer_semantic_return --output %t.ptrsemantic.mapping.csv --artifact %t.ptrsemantic.mapping.json 2>&1 | FileCheck %s --check-prefix=PTRSEM

// PTRSEM: graph returns unsupported pointer value for PnR mapping

// RUN: not loom-pnr-map --dfg-mlir %s --graph mem_pointer_return --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mem_pointer_return --output %t.ptrret.mapping.csv --artifact %t.ptrret.mapping.json 2>&1 | FileCheck %s --check-prefix=PTRRET

// PTRRET: graph returns unsupported pointer value for PnR mapping

module {
  dataflow.graph.func private @mem_route(%ctrl: none, %mem: memref<?xi32>,
                                         %idx: index, %rhs: i32)
      -> (none, i32) {
    %data, %done = dataflow.load %mem[%idx] %ctrl : memref<?xi32>
    %sum = arith.addi %data, %rhs : i32
    %synced = dataflow.sync %done : (none) -> none
    dataflow.graph.return %synced, %sum : none, i32
  }

  dataflow.graph.func private @mem_two_loads_one_port(
      %ctrl: none, %mem: memref<?xi32>, %lhs_idx: index, %rhs_idx: index)
      -> (none, i32) {
    %lhs, %lhs_done = dataflow.load %mem[%lhs_idx] %ctrl : memref<?xi32>
    %rhs, %rhs_done = dataflow.load %mem[%rhs_idx] %ctrl : memref<?xi32>
    %sum = arith.addi %lhs, %rhs : i32
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  fabric.module @mem_route_adg(%mgr : memref<?x!fabric.bits<32>>,
                               %addr : !fabric.bits<32>,
                               %ctrl : !fabric.bits<0>,
                               %rhs : !fabric.bits<32>) {
    %data, %done =
        fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl) store()
          [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
          : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
            -> (!fabric.bits<32>, !fabric.bits<0>)
    fabric.pe [spatial] (%lhs = %data : !fabric.bits<32>,
                         %right = %rhs : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>,
                %b = %right : !fabric.bits<32>) -> () {
        %sum = fabric.op [@arith.addi] (%a, %b)
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield
      }
    }
    fabric.pe [spatial] (%pc = %done : !fabric.bits<0>)
        -> !fabric.bits<0> {
      fabric.fu(%fc = %pc : !fabric.bits<0>) -> () {
        %synced = fabric.op [@dataflow.sync] (%fc)
                  {sw_configs = {bitmask = "1"}}
                  : (!fabric.bits<0>) -> !fabric.bits<0>
        fabric.yield
      }
    }
    fabric.yield
  }

  dataflow.graph.func private @mem_store_route(%ctrl: none, %mem: memref<?xi32>,
                                               %idx: index)
      -> (none) {
    %data, %done = dataflow.load %mem[%idx] %ctrl : memref<?xi32>
    %stored = dataflow.store %mem[%idx] %data %done : memref<?xi32>
    dataflow.graph.return %stored : none
  }

  dataflow.graph.func private @mem_gep_store(%ctrl: none, %src: !llvm.ptr,
                                             %dst: !llvm.ptr, %idx: index)
      -> (none) {
    %src_mem = builtin.unrealized_conversion_cast %src : !llvm.ptr to memref<?xi32>
    %dst_next = llvm.getelementptr inbounds|nuw %dst[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %dst_mem = builtin.unrealized_conversion_cast %dst_next : !llvm.ptr to memref<?xi32>
    %data, %done = dataflow.load %src_mem[%idx] %ctrl : memref<?xi32>
    %stored = dataflow.store %dst_mem[%idx] %data %done : memref<?xi32>
    dataflow.graph.return %stored : none
  }

  dataflow.graph.func private @mem_gep_pointer_store_value(
      %ctrl: none, %src: !llvm.ptr, %slot: !llvm.ptr) -> none {
    %src_next = llvm.getelementptr inbounds|nuw %src[4]
        : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.store %src_next, %slot : !llvm.ptr, !llvm.ptr
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @mem_scf_pointer_yield_bookkeeping(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %src: !llvm.ptr,
      %dst: !llvm.ptr, %idx: index) -> none {
    %0:2 = scf.for %iv = %lb to %ub step %step iter_args(%src_cur = %src,
                                                        %dst_cur = %dst)
        -> (!llvm.ptr, !llvm.ptr) : i32 {
      %src_next = llvm.getelementptr inbounds|nuw %src_cur[4]
          : (!llvm.ptr) -> !llvm.ptr, i8
      %src_mem = builtin.unrealized_conversion_cast %src_cur
          : !llvm.ptr to memref<?xf32>
      %data, %done = dataflow.load %src_mem[%idx] %ctrl : memref<?xf32>
      %dst_next = llvm.getelementptr inbounds|nuw %dst_cur[4]
          : (!llvm.ptr) -> !llvm.ptr, i8
      %dst_mem = builtin.unrealized_conversion_cast %dst_cur
          : !llvm.ptr to memref<?xf32>
      %stored = dataflow.store %dst_mem[%idx] %data %done : memref<?xf32>
      scf.yield %src_next, %dst_next : !llvm.ptr, !llvm.ptr
    } {loom.stream_step_kind = 0 : i32, loom.stream_predicate = 2 : i64}
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @mem_nested_scf_pointer_yield_bookkeeping(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %inner_ub: i32,
      %active: i1, %scale: i32, %out: !llvm.ptr, %in: !llvm.ptr) -> none {
    %0:2 = scf.for %iv = %lb to %ub step %step iter_args(%out_cur = %out,
                                                        %in_cur = %in)
        -> (!llvm.ptr, !llvm.ptr) : i32 {
      %1:2 = scf.if %active -> (!llvm.ptr, i32) {
        %11:2 = scf.for %j = %lb to %inner_ub step %step
            iter_args(%acc = %lb, %in_inner = %in_cur)
            -> (i32, !llvm.ptr) : i32 {
          %inner_next = llvm.getelementptr inbounds|nuw %in_inner[1]
              : (!llvm.ptr) -> !llvm.ptr, i8
          scf.yield %acc, %inner_next : i32, !llvm.ptr
        }
        %after_inner = llvm.getelementptr %in_cur[%inner_ub]
            : (!llvm.ptr, i32) -> !llvm.ptr, i8
        scf.yield %after_inner, %11#0 : !llvm.ptr, i32
      } else {
        scf.yield %in_cur, %lb : !llvm.ptr, i32
      }
      %in_mem = builtin.unrealized_conversion_cast %1#0
          : !llvm.ptr to memref<?xi8>
      %zero_in = dataflow.constant %ctrl {const_value = 0 : index} : index
      %data_i8, %loaded_i8 =
          dataflow.load %in_mem[%zero_in] %ctrl : memref<?xi8>
      %out_mem = builtin.unrealized_conversion_cast %out_cur
          : !llvm.ptr to memref<?xi8>
      %zero_out = dataflow.constant %ctrl {const_value = 0 : index} : index
      %stored = dataflow.store %out_mem[%zero_out] %data_i8 %loaded_i8
          : memref<?xi8>
      %out_next = llvm.getelementptr inbounds|nuw %out_cur[4]
          : (!llvm.ptr) -> !llvm.ptr, i8
      scf.yield %out_next, %1#0 : !llvm.ptr, !llvm.ptr
    } {loom.stream_step_kind = 0 : i32, loom.stream_predicate = 2 : i64}
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @mem_nested_scf_pointer_store_value(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %slot: !llvm.ptr,
      %src: !llvm.ptr) -> none {
    %0 = scf.for %iv = %lb to %ub step %step iter_args(%src_cur = %src)
        -> (!llvm.ptr) : i32 {
      %active = arith.cmpi slt, %iv, %ub : i32
      %1 = scf.if %active -> (!llvm.ptr) {
        %next = llvm.getelementptr inbounds|nuw %src_cur[1]
            : (!llvm.ptr) -> !llvm.ptr, i8
        scf.yield %next : !llvm.ptr
      } else {
        scf.yield %src_cur : !llvm.ptr
      }
      scf.yield %1 : !llvm.ptr
    } {loom.stream_step_kind = 0 : i32, loom.stream_predicate = 2 : i64}
    llvm.store %0, %slot : !llvm.ptr, !llvm.ptr
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @mem_scf_pointer_semantic_return(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %ptr: !llvm.ptr)
      -> (none, !llvm.ptr) {
    %0 = scf.for %iv = %lb to %ub step %step iter_args(%cur = %ptr)
        -> (!llvm.ptr) : i32 {
      %next = llvm.getelementptr inbounds|nuw %cur[4]
          : (!llvm.ptr) -> !llvm.ptr, i8
      scf.yield %next : !llvm.ptr
    } {loom.stream_step_kind = 0 : i32, loom.stream_predicate = 2 : i64}
    dataflow.graph.return values() streams() memories(%0 : !llvm.ptr)
        complete(%ctrl : none)
  }

  dataflow.graph.func private @mem_pointer_bookkeeping(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %bias: f32,
      %src: !llvm.ptr, %dst: !llvm.ptr) -> (none) {
    %src_mem = builtin.unrealized_conversion_cast %src : !llvm.ptr to memref<?xf32>
    %dst_mem = builtin.unrealized_conversion_cast %dst : !llvm.ptr to memref<?xf32>
    %idx, %rwc = dataflow.stream %lb, %ub, %step step add while slt : i32
    %addr = arith.index_cast %idx : i32 to index
    %src_cur = dataflow.carry %rwc, %src, %src_next : !llvm.ptr
    %dst_cur = dataflow.carry %rwc, %dst, %dst_next : !llvm.ptr
    %src_next = llvm.getelementptr inbounds|nuw %src_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %data, %done = dataflow.load %src_mem[%addr] %ctrl : memref<?xf32>
    %sum = arith.addf %data, %bias : f32
    %dst_next = llvm.getelementptr inbounds|nuw %dst_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %stored = dataflow.store %dst_mem[%addr] %sum %ctrl : memref<?xf32>
    %synced:2 = dataflow.sync %done, %stored : (none, none) -> (none, none)
    dataflow.graph.return %synced#0 : none
  }

  dataflow.graph.func private @mem_pointer_bookkeeping_return(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %bias: f32,
      %src: !llvm.ptr, %dst: !llvm.ptr) -> (none, !llvm.ptr) {
    %src_mem = builtin.unrealized_conversion_cast %src : !llvm.ptr to memref<?xf32>
    %dst_mem = builtin.unrealized_conversion_cast %dst : !llvm.ptr to memref<?xf32>
    %idx, %rwc = dataflow.stream %lb, %ub, %step step add while slt : i32
    %addr = arith.index_cast %idx : i32 to index
    %src_cur = dataflow.carry %rwc, %src, %src_next : !llvm.ptr
    %dst_cur = dataflow.carry %rwc, %dst, %dst_next : !llvm.ptr
    %src_next = llvm.getelementptr inbounds|nuw %src_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %data, %done = dataflow.load %src_mem[%addr] %ctrl : memref<?xf32>
    %sum = arith.addf %data, %bias : f32
    %dst_next = llvm.getelementptr inbounds|nuw %dst_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %stored = dataflow.store %dst_mem[%addr] %sum %ctrl : memref<?xf32>
    %synced:2 = dataflow.sync %done, %stored : (none, none) -> (none, none)
    dataflow.graph.return values() streams() memories(%dst_cur : !llvm.ptr)
        complete(%synced#0 : none)
  }

  dataflow.graph.func private @mem_gep_bookkeeping_return(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %bias: f32,
      %src: !llvm.ptr, %dst: !llvm.ptr) -> (none, !llvm.ptr) {
    %src_mem = builtin.unrealized_conversion_cast %src : !llvm.ptr to memref<?xf32>
    %dst_mem = builtin.unrealized_conversion_cast %dst : !llvm.ptr to memref<?xf32>
    %idx, %rwc = dataflow.stream %lb, %ub, %step step add while slt : i32
    %addr = arith.index_cast %idx : i32 to index
    %src_cur = dataflow.carry %rwc, %src, %src_next : !llvm.ptr
    %dst_cur = dataflow.carry %rwc, %dst, %dst_next : !llvm.ptr
    %src_next = llvm.getelementptr inbounds|nuw %src_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %data, %done = dataflow.load %src_mem[%addr] %ctrl : memref<?xf32>
    %sum = arith.addf %data, %bias : f32
    %dst_next = llvm.getelementptr inbounds|nuw %dst_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %stored = dataflow.store %dst_mem[%addr] %sum %ctrl : memref<?xf32>
    %synced:2 = dataflow.sync %done, %stored : (none, none) -> (none, none)
    dataflow.graph.return values() streams() memories(%dst_next : !llvm.ptr)
        complete(%synced#0 : none)
  }

  dataflow.graph.func private @control_mux_needs_control_resource(
      %ctrl: none, %sel: i1) -> none {
    %done = dataflow.mux %sel, %ctrl, %ctrl : (i1, none, none) -> none
    dataflow.graph.return %done : none
  }

  dataflow.graph.func private @predicate_and_maps_to_transport_andi(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %lhs0: i32, %rhs0: i32,
      %lhs1: i32, %rhs1: i32, %mem: memref<?xf32>) -> (none, f32) {
    %idx, %rwc = dataflow.stream %lb, %ub, %step step add while slt : i32
    %idx_as_index = arith.index_cast %idx : i32 to index
    %p0 = arith.cmpi sgt, %lhs0, %rhs0 : i32
    %p1 = arith.cmpi slt, %lhs1, %rhs1 : i32
    %both = arith.andi %p0, %p1 : i1
    %zero = dataflow.constant %ctrl {const_value = 0 : index} : index
    %addr = arith.select %both, %idx_as_index, %zero : index
    %data, %done = dataflow.load %mem[%addr] %ctrl : memref<?xf32>
    dataflow.graph.return %done, %data : none, f32
  }

  dataflow.graph.func private @llvm_load_pointer(%ctrl: none, %ptr: !llvm.ptr,
                                                 %rhs: i32) -> (none, i32) {
    %next = llvm.getelementptr inbounds|nuw %ptr[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %data = llvm.load %next {alignment = 4 : i64} : !llvm.ptr -> i32
    %sum = arith.addi %data, %rhs : i32
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  dataflow.graph.func private @llvm_select_pointer_map(
      %ctrl: none, %lhs_value: i32, %rhs_value: i32, %lhs: !llvm.ptr,
      %rhs: !llvm.ptr, %bias: i32) -> (none, i32) {
    %pred = arith.cmpi sgt, %lhs_value, %rhs_value : i32
    %selected = llvm.select %pred, %lhs, %rhs : i1, !llvm.ptr
    %data = llvm.load %selected {alignment = 4 : i64} : !llvm.ptr -> i32
    %sum = arith.addi %data, %bias : i32
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  dataflow.graph.func private @llvm_select_pointer_wide_cmp_map(
      %ctrl: none, %iv: i64, %pivot: i64, %limit: i64, %lhs: !llvm.ptr,
      %rhs: !llvm.ptr, %bias: f32) -> (none, f32) {
    %same = arith.cmpi eq, %iv, %pivot : i64
    %value = scf.if %same -> (f32) {
      scf.yield %bias : f32
    } else {
      %before = arith.cmpi ult, %iv, %limit : i64
      %selected = llvm.select %before, %lhs, %rhs : i1, !llvm.ptr
      %data = llvm.load %selected {alignment = 4 : i64} : !llvm.ptr -> f32
      scf.yield %data : f32
    }
    %sum = arith.addf %value, %bias : f32
    dataflow.graph.return %ctrl, %sum : none, f32
  }

  dataflow.graph.func private @llvm_store_pointer(%ctrl: none, %src: !llvm.ptr,
                                                  %dst: !llvm.ptr) -> none {
    %data = llvm.load %src {alignment = 4 : i64} : !llvm.ptr -> f32
    %negated = llvm.fneg %data : f32
    llvm.store %negated, %dst {alignment = 4 : i64} : f32, !llvm.ptr
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @constant_addr_load_store(
      %ctrl: none, %src: memref<?xf32>, %dst: memref<?xf32>) -> none {
    %idx = dataflow.constant %ctrl {const_value = 0 : index} : index
    %data, %loaded = dataflow.load %src[%idx] %ctrl : memref<?xf32>
    %negated = llvm.fneg %data : f32
    %stored = dataflow.store %dst[%idx] %negated %ctrl : memref<?xf32>
    %done:2 = dataflow.sync %loaded, %stored : (none, none) -> (none, none)
    dataflow.graph.return %done#0 : none
  }

  dataflow.graph.func private @cfft_red3_fmul_pair(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %twiddle: f32,
      %buf: !llvm.ptr) -> none
      attributes {input_segments = array<i32: 4, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %mem = builtin.unrealized_conversion_cast %buf : !llvm.ptr to memref<?xf32>
    %one = dataflow.constant %ctrl {const_value = 1 : index} : index
    %iv, %phase = dataflow.stream %lb, %ub, %step step add while slt : i32
    %execution = dataflow.carry %phase, %ctrl, %iteration_done : none
    %execution_lanes:2 = dataflow.demux %phase, %execution
        : (i1, none) -> (none, none)
    %one_each = dataflow.invariant %phase, %one : index
    %one_cond, %one_active = dataflow.gate %phase, %one_each : index
    %scale_each = dataflow.invariant %phase, %twiddle : f32
    %scale_cond, %scale = dataflow.gate %phase, %scale_each : f32
    %read_frontier = dataflow.carry %phase, %ctrl, %iteration_done : none
    %write_frontier = dataflow.carry %phase, %ctrl, %iteration_done : none
    %read_lanes:2 = dataflow.demux %phase, %read_frontier
        : (i1, none) -> (none, none)
    %write_lanes:2 = dataflow.demux %phase, %write_frontier
        : (i1, none) -> (none, none)
    %index = arith.index_cast %iv : i32 to index
    %base = arith.addi %index, %index : index
    %next = arith.addi %base, %one_active : index
    %load_ready:2 = dataflow.sync %execution_lanes#1, %read_lanes#1
        : (none, none) -> (none, none)
    %store_ready:2 = dataflow.sync %write_lanes#1, %loaded0
        : (none, none) -> (none, none)
    %data0, %loaded0 = dataflow.load %mem[%base] %load_ready#0
        : memref<?xf32>
    %scaled0 = arith.mulf %scale, %data0 : f32
    %stored0 = dataflow.store %mem[%base] %scaled0 %store_ready#0
        : memref<?xf32>
    %data1, %loaded1 = dataflow.load %mem[%next] %stored0 : memref<?xf32>
    %neg = llvm.fneg %data1 : f32
    %scaled1 = arith.mulf %scale, %neg : f32
    %iteration_done = dataflow.store %mem[%next] %scaled1 %loaded1
        : memref<?xf32>
    dataflow.graph.return values() streams() memories()
        complete(%execution_lanes#0, %write_lanes#0 : none, none)
  }

  dataflow.graph.func private @mem_pointer_semantic_return(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %src: !llvm.ptr)
      -> (none, !llvm.ptr, i32) {
    %idx, %rwc = dataflow.stream %lb, %ub, %step step add while slt : i32
    %src_cur = dataflow.carry %rwc, %src, %src_next : !llvm.ptr
    %src_next = llvm.getelementptr inbounds|nuw %src_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
    %bits = builtin.unrealized_conversion_cast %src_cur : !llvm.ptr to i32
    %sum = arith.addi %bits, %lb : i32
    dataflow.graph.return values(%sum : i32) streams()
        memories(%src_cur : !llvm.ptr) complete(%ctrl : none)
  }

  dataflow.graph.func private @mem_pointer_return(%ctrl: none, %ptr: !llvm.ptr)
      -> (none, !llvm.ptr) {
    dataflow.graph.return values() streams() memories(%ptr : !llvm.ptr)
        complete(%ctrl : none)
  }

  fabric.module @mem_store_route_adg(%mgr : memref<?x!fabric.bits<32>>,
                                     %addr : !fabric.bits<32>,
                                     %ctrl : !fabric.bits<0>) {
    %addr_to_load, %addr_to_store = fabric.switch [spatial] %addr
        [{connectivity_table = ["1", "1"]}]
        : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
    %sub, %data, %done =
        fabric.mem [spatial] mgr(%mgr) load(%addr_to_load, %ctrl) store()
          [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
          : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
            -> (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
    %stored =
        fabric.mem [spatial] mgr(%sub) load()
            store(%addr_to_store, %data, %done)
          [{load_group_size = 0 : i32, store_group_size = 1 : i32}]
          : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<32>,
             !fabric.bits<0>) -> !fabric.bits<0>
    fabric.yield
  }

  fabric.module @data_mux_only_adg(%sel_src : !fabric.bits<32>,
                                   %lhs : !fabric.bits<32>,
                                   %rhs : !fabric.bits<32>) {
    %selected = fabric.pe [spatial] (%pa = %sel_src : !fabric.bits<32>,
                                     %pb = %lhs : !fabric.bits<32>,
                                     %pc = %rhs : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%sel = %pa : !fabric.bits<32> to !fabric.bits<1>,
                %false_lane = %pb : !fabric.bits<32>,
                %true_lane = %pc : !fabric.bits<32>)
          -> !fabric.bits<32> {
        %out = fabric.op [@dataflow.mux] (%sel, %false_lane, %true_lane)
            : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>)
              -> !fabric.bits<32>
        fabric.yield %out : !fabric.bits<32>
      }
    }
    fabric.yield
  }
}
