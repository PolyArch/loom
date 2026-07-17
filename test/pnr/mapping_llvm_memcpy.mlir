// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: FileCheck %s --check-prefix=STRUCTURED-LOWERED < %t.lowered.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph pointer_memcpy_structured_if --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload pointer_memcpy_structured_if --output %t.structured.mapping.csv --artifact %t.structured.mapping.json
// RUN: FileCheck %s --check-prefix=STRUCTURED-CSV < %t.structured.mapping.csv
// RUN: FileCheck %s --check-prefix=STRUCTURED-JSON < %t.structured.mapping.json
// RUN: loom-adg-builder-test --shared-memory-reduction --output %t.hardware.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph lowered_copy --hardware-mlir %t.hardware.mlir --hardware shared_memory_reduction_adg --workload lowered_copy --output %t.copy.mapping.csv --artifact %t.copy.mapping.json
// RUN: FileCheck %s --check-prefix=COPY-CSV < %t.copy.mapping.csv
// RUN: FileCheck %s --check-prefix=COPY-JSON < %t.copy.mapping.json

// STRUCTURED-LOWERED-LABEL: dataflow.graph private @pointer_memcpy_structured_if
// STRUCTURED-LOWERED-NOT: llvm.intr.memcpy
// STRUCTURED-LOWERED-NOT: scf.if
// STRUCTURED-LOWERED-NOT: scf.for
// STRUCTURED-LOWERED: dataflow.demux
// STRUCTURED-LOWERED: dataflow.carry
// STRUCTURED-LOWERED: arith.cmpi ult
// STRUCTURED-LOWERED: dataflow.gate
// STRUCTURED-LOWERED: dataflow.load
// STRUCTURED-LOWERED: dataflow.store
// STRUCTURED-LOWERED-NOT: llvm.intr.memcpy
// STRUCTURED-LOWERED: dataflow.graph.return

// The recursively lowered workload is larger than this fixed ADG. Keep the
// resource-pressure result explicit instead of silently replacing the mapping
// probe or widening the hardware fixture.
// STRUCTURED-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// STRUCTURED-CSV-NEXT: pointer_memcpy_structured_if,shared_reduction_adg,pointer_memcpy_structured_if__pointer_memcpy_structured_if__shared_reduction_adg,22,9,19,16,fail,missing hardware resource for software op dataflow.demux
// STRUCTURED-CSV-SAME: operation=dataflow.carry required=5 available=5 placed=1 missing=4
// STRUCTURED-CSV-SAME: operation=dataflow.demux required=12 available=3 placed=3 missing=9
// STRUCTURED-CSV-SAME: operation=dataflow.gate required=4 available=4 placed=3 missing=1
// STRUCTURED-CSV-SAME: operation=dataflow.mux required=2 available=1 placed=1 missing=1
// STRUCTURED-CSV-SAME: operation=dataflow.sync required=3 available=2 placed=2 missing=1

// STRUCTURED-JSON-DAG: "status": "fail"
// STRUCTURED-JSON-DAG: "placed_records": 22
// STRUCTURED-JSON-DAG: "routed_edges": 9
// STRUCTURED-JSON-DAG: "unrouted_edges": 19
// STRUCTURED-JSON-DAG: "unplaced_records": 16
// STRUCTURED-JSON-DAG: "operation": "dataflow.load"
// STRUCTURED-JSON-DAG: "operation": "dataflow.store"
// STRUCTURED-JSON-DAG: "operation": "dataflow.carry"
// STRUCTURED-JSON-DAG: "required": 5
// STRUCTURED-JSON-DAG: "operation": "dataflow.demux"
// STRUCTURED-JSON-DAG: "required": 12
// STRUCTURED-JSON-DAG: "operation": "dataflow.sync"
// STRUCTURED-JSON-DAG: "required": 3

// COPY-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// COPY-CSV-NEXT: lowered_copy,shared_memory_reduction_adg,lowered_copy__lowered_copy__shared_memory_reduction_adg,2,2,0,0,pass

// COPY-JSON-DAG: "status": "pass"
// COPY-JSON-DAG: "operation": "dataflow.load"
// COPY-JSON-DAG: "resource_kind": "fabric.mem.load"
// COPY-JSON-DAG: "operation": "dataflow.store"
// COPY-JSON-DAG: "resource_kind": "fabric.mem.store"
// COPY-JSON-DAG: "edge_ref": "dataflow.load#0.result0->dataflow.store#0.operand2"
// COPY-JSON-DAG: "edge_ref": "dataflow.load#0.result1->dataflow.store#0.operand3"
// COPY-JSON-NOT: "fabric.mem.copy"
// COPY-JSON-NOT: "memory_copy_binding"
// COPY-JSON-NOT: "llvm.intr.memcpy"

// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph lowered_two_copies_port_no_reuse --hardware-mlir %S/mapping_mem_route.mlir --hardware mem_store_route_adg --workload lowered_two_copies_port_no_reuse --output %t.noreuse.mapping.csv --artifact %t.noreuse.mapping.json
// RUN: FileCheck %s --check-prefix=NOREUSE-CSV < %t.noreuse.mapping.csv
// RUN: FileCheck %s --check-prefix=NOREUSE-JSON < %t.noreuse.mapping.json

// NOREUSE-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// NOREUSE-CSV-NEXT: lowered_two_copies_port_no_reuse,mem_store_route_adg,lowered_two_copies_port_no_reuse__lowered_two_copies_port_no_reuse__mem_store_route_adg,2,1,0,3,fail,missing hardware resource for software op dataflow.store

// NOREUSE-JSON-DAG: "status": "fail"
// NOREUSE-JSON-DAG: "unplaced_records": 3
// NOREUSE-JSON-DAG: "operation": "dataflow.load"
// NOREUSE-JSON-DAG: "resource_kind": "fabric.mem.load"
// NOREUSE-JSON-DAG: "operation": "dataflow.store"
// NOREUSE-JSON-DAG: "resource_kind": "fabric.mem.store"
// NOREUSE-JSON-NOT: "fabric.mem.copy"
// NOREUSE-JSON-NOT: "memory_copy_binding"
// NOREUSE-JSON-NOT: "llvm.intr.memcpy"

module {
  dataflow.graph private @lowered_copy(
      %ctrl: none, %src_index: index, %dst_index: index,
      %src: memref<?xi8>, %dst: memref<?xi8>) -> ()
      attributes {input_segments = array<i32: 2, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
    %data, %loaded = dataflow.load %src[%src_index] %ctrl : memref<?xi8>
    %stored = dataflow.store %dst[%dst_index] %data %loaded : memref<?xi8>
    dataflow.graph.return values() streams() memories()
        complete(%stored : none)
  }

  dataflow.graph private @pointer_memcpy_structured_if(
      %ctrl: none, %copy_bytes: i32, %do_copy: i1, %src_offset: i32,
      %dst_offset: i32, %src: !llvm.ptr, %dst: !llvm.ptr) -> ()
      attributes {input_segments = array<i32: 4, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
    scf.if %do_copy {
      %src_at = llvm.getelementptr %src[%src_offset]
          : (!llvm.ptr, i32) -> !llvm.ptr, i8
      %dst_at = llvm.getelementptr %dst[%dst_offset]
          : (!llvm.ptr, i32) -> !llvm.ptr, i8
      "llvm.intr.memcpy"(%dst_at, %src_at, %copy_bytes)
        <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
           isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    }
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph private @lowered_two_copies_port_no_reuse(
      %ctrl: none, %src0: index, %dst0: index, %src1: index, %dst1: index,
      %mem: memref<?xi32>) -> ()
      attributes {input_segments = array<i32: 4, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %data0, %load0_done = dataflow.load %mem[%src0] %ctrl : memref<?xi32>
    %data1, %load1_done = dataflow.load %mem[%src1] %ctrl : memref<?xi32>
    %store0_done =
        dataflow.store %mem[%dst0] %data0 %load0_done : memref<?xi32>
    %store1_done =
        dataflow.store %mem[%dst1] %data1 %load1_done : memref<?xi32>
    dataflow.graph.return values() streams() memories()
        complete(%store0_done, %store1_done : none, none)
  }
}
