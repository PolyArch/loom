// RUN: loom-adg-builder-test --shared-reduction --output %t.shared.hardware.mlir
// RUN: loom-pnr-map --dfg-mlir %s --graph mem_route --hardware-mlir %s --hardware mem_route_adg --workload mem_route --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: mem_route,mem_route_adg,mem_route__mem_route__mem_route_adg,3,3,0,0,pass

// JSON-DAG: "status": "pass"
// JSON-DAG: "hardware": "mem_route_adg::mem.load#0"
// JSON-DAG: "segment_kind": "resource_edge"
// JSON-DAG: "source_endpoint": "mem_route_adg::mem.load#0.result0"
// JSON-DAG: "sink_endpoint": "mem_route_adg::fabric.op#1.operand0"
// JSON-DAG: "source_endpoint": "mem_route_adg::mem.load#0.result1"
// JSON-DAG: "sink_endpoint": "mem_route_adg::fabric.op#2.operand0"
// JSON-NOT: ".out"
// JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph mem_route --hardware-mlir %t.shared.hardware.mlir --hardware shared_reduction_adg --workload mem_route_shared_sync_prefix --output %t.shared-sync.mapping.csv --artifact %t.shared-sync.mapping.json
// RUN: FileCheck %s --check-prefix=SHARED-SYNC-CSV < %t.shared-sync.mapping.csv
// RUN: FileCheck %s --check-prefix=SHARED-SYNC-JSON < %t.shared-sync.mapping.json

// SHARED-SYNC-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// SHARED-SYNC-CSV-NEXT: mem_route_shared_sync_prefix,shared_reduction_adg,mem_route_shared_sync_prefix__mem_route__shared_reduction_adg,3,3,0,0,pass

// SHARED-SYNC-JSON-DAG: "status": "pass"
// SHARED-SYNC-JSON-DAG: "operation": "dataflow.sync"
// SHARED-SYNC-JSON-DAG: "edge_ref": "dataflow.load#0.result1->dataflow.sync#0.operand0"
// SHARED-SYNC-JSON-NOT: ".out"
// SHARED-SYNC-JSON-NOT: ".in"

// RUN: loom-pnr-map --dfg-mlir %s --graph mem_two_loads_one_port --hardware-mlir %s --hardware mem_route_adg --workload mem_two_loads_one_port --output %t.twoload.mapping.csv --artifact %t.twoload.mapping.json
// RUN: FileCheck %s --check-prefix=TWOLOAD-CSV < %t.twoload.mapping.csv
// RUN: FileCheck %s --check-prefix=TWOLOAD-JSON < %t.twoload.mapping.json

// TWOLOAD-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// TWOLOAD-CSV-NEXT: mem_two_loads_one_port,mem_route_adg,mem_two_loads_one_port__mem_two_loads_one_port__mem_route_adg,4,4,0,1,fail,missing hardware resource for software op dataflow.load

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

// RUN: loom-pnr-map --dfg-mlir %s --graph predicate_and_maps_to_transport_andi --hardware-mlir %t.shared.hardware.mlir --hardware shared_reduction_adg --workload predicate_and --output %t.predand.mapping.csv --artifact %t.predand.mapping.json
// RUN: FileCheck %s --check-prefix=PREDAND-JSON < %t.predand.mapping.json

// PREDAND-JSON-DAG: "status": "pass"
// PREDAND-JSON-DAG: "operation": "arith.andi"
// PREDAND-JSON-DAG: "edge_ref": "arith.cmpi#0.result0->arith.andi#0.operand0"
// PREDAND-JSON-DAG: "edge_ref": "arith.cmpi#1.result0->arith.andi#0.operand1"
// PREDAND-JSON-DAG: "edge_ref": "arith.andi#0.result0->arith.select#0.operand0"
// PREDAND-JSON-NOT: "missing hardware resource for software op arith.andi"

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

// RUN: loom-pnr-map --dfg-mlir %s --graph mem_pointer_return --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload mem_pointer_return --output %t.ptrret.mapping.csv --artifact %t.ptrret.mapping.json
// RUN: FileCheck %s --check-prefix=PTRRET-CSV < %t.ptrret.mapping.csv
// RUN: FileCheck %s --check-prefix=PTRRET-JSON < %t.ptrret.mapping.json

// PTRRET-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// PTRRET-CSV-NEXT: mem_pointer_return,shared_reduction_adg,mem_pointer_return__mem_pointer_return__shared_reduction_adg,0,0,0,0,pass

// PTRRET-JSON-DAG: "status": "pass"
// PTRRET-JSON-DAG: "placements": []
// PTRRET-JSON-DAG: "routes": []

module {
  dataflow.graph private @mem_route(%ctrl: none, %idx: index, %rhs: i32,
                                         %mem: memref<?xi32>)
      -> (i32)
      attributes {input_segments = array<i32: 2, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %data, %done = dataflow.load %mem[%idx] %ctrl : memref<?xi32>
    %sum = arith.addi %data, %rhs : i32
    %published:2 = dataflow.sync %done, %sum
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%published#1 : i32) streams() memories()
        complete(%published#0 : none)
  }

  dataflow.graph private @mem_two_loads_one_port(
      %ctrl: none, %lhs_idx: index, %rhs_idx: index, %mem: memref<?xi32>)
      -> (i32)
      attributes {input_segments = array<i32: 2, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %lhs, %lhs_done = dataflow.load %mem[%lhs_idx] %ctrl : memref<?xi32>
    %rhs, %rhs_done = dataflow.load %mem[%rhs_idx] %ctrl : memref<?xi32>
    %sum = arith.addi %lhs, %rhs : i32
    %effects:2 = dataflow.sync %lhs_done, %rhs_done
        : (none, none) -> (none, none)
    %published:2 = dataflow.sync %effects#0, %sum
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%published#1 : i32) streams() memories()
        complete(%published#0 : none)
  }

  fabric.module @mem_route_adg(%mgr : memref<?x!fabric.bits<32>>,
                               %addr : !fabric.bits<32>,
                               %ctrl : !fabric.bits<0>,
                               %rhs : !fabric.bits<32>) {
    %load_ctrl, %sync_ctrl = fabric.switch [spatial] %ctrl
        [{connectivity_table = ["1", "1"]}]
        : (!fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>)
    %data, %done =
        fabric.mem [spatial] mgr(%mgr) load(%addr, %load_ctrl) store()
          [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
          : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
            -> (!fabric.bits<32>, !fabric.bits<0>)
    %done_to_publish, %done_to_join = fabric.switch [spatial] %done
        [{connectivity_table = ["1", "1"]}]
        : (!fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>)
    %joined = fabric.pe [spatial] (
        %pa = %done_to_join : !fabric.bits<0>,
        %pb = %sync_ctrl : !fabric.bits<0>) -> !fabric.bits<0> {
      fabric.fu(%a = %pa : !fabric.bits<0>,
                %b = %pb : !fabric.bits<0>) -> !fabric.bits<0> {
        %done0, %done1 = fabric.op [@dataflow.sync] (%a, %b)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<0>)
              -> (!fabric.bits<0>, !fabric.bits<0>)
        fabric.yield %done0 : !fabric.bits<0>
      }
    }
    %publish_ctrl = fabric.switch [spatial] %done_to_publish, %joined
        [{connectivity_table = ["11"]}]
        : (!fabric.bits<0>, !fabric.bits<0>) -> !fabric.bits<0>
    fabric.pe [spatial] (%lhs = %data : !fabric.bits<32>,
                         %right = %rhs : !fabric.bits<32>,
                         %pc = %publish_ctrl : !fabric.bits<0> to !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%a = %lhs : !fabric.bits<32>,
                %b = %right : !fabric.bits<32>,
                %token = %pc : !fabric.bits<32> to !fabric.bits<0>) -> () {
        %sum = fabric.op [@arith.addi] (%a, %b)
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        %retired, %published = fabric.op [@dataflow.sync] (%token, %sum)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<32>)
              -> (!fabric.bits<0>, !fabric.bits<32>)
        fabric.yield
      }
    }
    fabric.yield
  }

  dataflow.graph private @mem_store_route(%ctrl: none, %idx: index,
                                               %mem: memref<?xi32>) -> ()
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %data, %done = dataflow.load %mem[%idx] %ctrl : memref<?xi32>
    %stored = dataflow.store %mem[%idx] %data %done : memref<?xi32>
    dataflow.graph.return %stored : none
  }










  dataflow.graph private @control_mux_needs_control_resource(
      %ctrl: none, %sel: i1) -> () {
    %done = dataflow.mux %sel, %ctrl, %ctrl : (i1, none, none) -> none
    dataflow.graph.return %done : none
  }

  dataflow.graph private @predicate_and_maps_to_transport_andi(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %lhs0: i32, %rhs0: i32,
      %lhs1: i32, %rhs1: i32, %mem: memref<?xf32>) -> (f32)
      attributes {input_segments = array<i32: 7, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %idx, %rwc = dataflow.stream %lb, %ub, %step step add while slt : i32
    %idx_as_index = arith.index_cast %idx : i32 to index
    %p0 = arith.cmpi sgt, %lhs0, %rhs0 : i32
    %p1 = arith.cmpi slt, %lhs1, %rhs1 : i32
    %both = arith.andi %p0, %p1 : i1
    %zero = dataflow.constant %ctrl {const_value = 0 : index} : index
    %addr = arith.select %both, %idx_as_index, %zero : index
    %data, %done = dataflow.load %mem[%addr] %ctrl : memref<?xf32>
    %phase_lanes:2 = dataflow.demux %rwc, %ctrl
        : (i1, none) -> (none, none)
    %published:2 = dataflow.sync %done, %data
        : (none, f32) -> (none, f32)
    dataflow.graph.return values(%published#1 : f32) streams() memories()
        complete(%phase_lanes#0, %published#0 : none, none)
  }





  dataflow.graph private @constant_addr_load_store(
      %ctrl: none, %src: memref<?xf32>, %dst: memref<?xf32>) -> ()
      attributes {input_segments = array<i32: 0, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
    %idx = dataflow.constant %ctrl {const_value = 0 : index} : index
    %data, %loaded = dataflow.load %src[%idx] %ctrl : memref<?xf32>
    %negated = llvm.fneg %data : f32
    %stored = dataflow.store %dst[%idx] %negated %ctrl : memref<?xf32>
    %done:2 = dataflow.sync %loaded, %stored : (none, none) -> (none, none)
    dataflow.graph.return %done#0 : none
  }

  dataflow.graph private @cfft_red3_fmul_pair(
      %ctrl: none, %lb: i32, %ub: i32, %step: i32, %twiddle: f32,
      %buf: !llvm.ptr) -> ()
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


  dataflow.graph private @mem_pointer_return(%ctrl: none, %ptr: !llvm.ptr)
      -> (!llvm.ptr)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 1>} {
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
