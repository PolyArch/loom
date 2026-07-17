// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --shared-memory-reduction --output %t.dir/shared-memory-reduction.mlir
// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph spmm_computed_address --hardware-mlir %t.dir/shared-memory-reduction.mlir --hardware shared_memory_reduction_adg --workload spmm_computed_address --output %t.csv --artifact %t.json
// RUN: FileCheck %s --check-prefix=CSV < %t.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: spmm_computed_address,shared_memory_reduction_adg,spmm_computed_address__spmm_computed_address__shared_memory_reduction_adg,40,35,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "software": "dataflow.constant#2"
// JSON-DAG: "software": "arith.index_cast#1"
// JSON-DAG: "edge_ref": "dataflow.constant#2.result0->arith.index_cast#1.operand0"
// JSON-DAG: "edge_ref": "arith.index_cast#1.result0->arith.addi#1.operand1"
// JSON-DAG: "segment_kind": "buffer"
// JSON-DAG: "hardware_ref": "shared_memory_reduction_adg::fabric.switch#0"
// JSON-DAG: "edge_ref": "arith.index_cast#0.result0->dataflow.load#0.operand1"
// JSON-DAG: "sink_endpoint": "shared_memory_reduction_adg::mem.load#0.operand0"
// JSON-DAG: "unrouted_edges": 0
// JSON-DAG: "unrouted_edge_details": []
// JSON-DAG: "status": "pass"

module {
  dataflow.graph.func private @spmm_computed_address(
      %arg0: none, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr,
      %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: i32, %arg7: i32) -> none {
    %0 = dataflow.constant %arg0 {const_value = 0 : i32} : i32
    %1 = dataflow.constant %arg0 {const_value = 0 : i64} : i64
    %2 = dataflow.constant %arg0 {const_value = 1 : i64} : i64
    %3 = arith.muli %arg7, %arg6 : i32
    %4 = arith.cmpi eq, %3, %0 : i32
    scf.if %4 {
    } else {
      %6 = builtin.unrealized_conversion_cast %arg5
          : !llvm.ptr to memref<?xi32>
      %7 = dataflow.constant %arg0 {const_value = 0 : index} : index
      %8 = dataflow.constant %arg0 {const_value = 1 : index} : index
      %9 = arith.index_cast %arg7 : i32 to index
      %10 = arith.index_cast %arg6 : i32 to index
      %11 = arith.muli %9, %10 : index
      scf.for %arg8 = %7 to %11 step %8 {
        %12 = dataflow.constant %arg0 {const_value = 0 : i32} : i32
        %13 = dataflow.store %6[%arg8] %12 %arg0 : memref<?xi32>
      }
    }
    %5 = arith.cmpi eq, %arg6, %0 : i32
    scf.if %5 {
    } else {
      %6 = arith.cmpi eq, %arg7, %0 : i32
      %7 = llvm.zext %arg6 : i32 to i64
      %8 = llvm.zext %arg7 : i32 to i64
      %9 = scf.while (%arg8 = %1) : (i64) -> i64 {
        %10 = builtin.unrealized_conversion_cast %arg3
            : !llvm.ptr to memref<?xi32>
        %11 = arith.index_cast %arg8 : i64 to index
        %data, %done = dataflow.load %10[%11] %arg0 : memref<?xi32>
        %12 = arith.addi %arg8, %2 : i64
        %13 = builtin.unrealized_conversion_cast %arg3
            : !llvm.ptr to memref<?xi32>
        %14 = arith.index_cast %arg8 : i64 to index
        %15 = arith.index_cast %2 : i64 to index
        %16 = arith.addi %14, %15 : index
        %data_0, %done_1 = dataflow.load %13[%16] %arg0 : memref<?xi32>
        %17 = arith.cmpi ult, %data, %data_0 : i32
        scf.if %17 {
          %19 = llvm.trunc %arg8 overflow<nuw> : i64 to i32
          %20 = llvm.zext %data : i32 to i64
          %21 = llvm.zext %data_0 : i32 to i64
          scf.for %arg9 = %20 to %21 step %2 : i64 {
            %22 = builtin.unrealized_conversion_cast %arg1
                : !llvm.ptr to memref<?xi32>
            %23 = arith.index_cast %arg9 : i64 to index
            %data_2, %done_3 = dataflow.load %22[%23] %arg0
                : memref<?xi32>
            scf.if %6 {
            } else {
              %24 = builtin.unrealized_conversion_cast %arg2
                  : !llvm.ptr to memref<?xi32>
              %25 = arith.index_cast %arg9 : i64 to index
              %data_4, %done_5 = dataflow.load %24[%25] %arg0
                  : memref<?xi32>
              scf.for %arg10 = %1 to %8 step %2 : i64 {
                %26 = llvm.trunc %arg10 overflow<nuw> : i64 to i32
                %27 = builtin.unrealized_conversion_cast %arg4
                    : !llvm.ptr to memref<?xi32>
                %28 = arith.index_cast %data_4 : i32 to index
                %29 = arith.index_cast %arg7 : i32 to index
                %30 = arith.muli %28, %29 : index
                %31 = arith.index_cast %26 : i32 to index
                %32 = arith.addi %30, %31 : index
                %data_6, %done_7 = dataflow.load %27[%32] %arg0
                    : memref<?xi32>
                %33 = arith.muli %data_6, %data_2 : i32
                %34 = llvm.trunc %arg10 overflow<nuw> : i64 to i32
                %35 = builtin.unrealized_conversion_cast %arg5
                    : !llvm.ptr to memref<?xi32>
                %36 = arith.index_cast %arg7 : i32 to index
                %37 = arith.index_cast %19 : i32 to index
                %38 = arith.muli %36, %37 : index
                %39 = arith.index_cast %34 : i32 to index
                %40 = arith.addi %38, %39 : index
                %data_8, %done_9 = dataflow.load %35[%40] %arg0
                    : memref<?xi32>
                %41 = arith.addi %data_8, %33 : i32
                %42 = builtin.unrealized_conversion_cast %arg5
                    : !llvm.ptr to memref<?xi32>
                %43 = arith.index_cast %arg7 : i32 to index
                %44 = arith.index_cast %19 : i32 to index
                %45 = arith.muli %43, %44 : index
                %46 = arith.index_cast %34 : i32 to index
                %47 = arith.addi %45, %46 : index
                %48 = dataflow.store %42[%47] %41 %arg0 : memref<?xi32>
              }
            }
          }
        }
        %18 = arith.cmpi ne, %12, %7 : i64
        scf.condition(%18) %12 : i64
      } do {
      ^bb0(%arg8: i64):
        scf.yield %arg8 : i64
      }
    }
    dataflow.graph.return %arg0 : none
  }
}
