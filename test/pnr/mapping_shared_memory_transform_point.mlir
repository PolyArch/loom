// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --shared-memory-reduction --output %t.dir/shared-memory-reduction.mlir
// RUN: loom-pnr-map --dfg-mlir %s --graph transform_point --hardware-mlir %t.dir/shared-memory-reduction.mlir --hardware shared_memory_reduction_adg --workload transform_point --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: transform_point,shared_memory_reduction_adg,transform_point__transform_point__shared_memory_reduction_adg,25,38,0,0,pass

// JSON-DAG: "status": "pass"
// JSON-DAG: "hardware": "shared_memory_reduction_adg"
// JSON-DAG: "operation": "llvm.intr.fmuladd"
// JSON-DAG: "operation": "arith.addf"
// JSON-DAG: "operation": "arith.mulf"
// JSON-DAG: "operation": "dataflow.load"
// JSON-DAG: "operation": "dataflow.store"
// JSON-DAG: "operation": "dataflow.sync"
// JSON-NOT: "resource_pressure"

module {
  dataflow.graph.func private @transform_point(
      %ctrl: none,
      %stride: i32,
      %input_points: !llvm.ptr,
      %y_offset: i32,
      %z_offset: i32,
      %m01: f32,
      %m00: f32,
      %m02: f32,
      %tx: f32,
      %output_points: !llvm.ptr,
      %m11: f32,
      %m10: f32,
      %m12: f32,
      %ty: f32,
      %m21: f32,
      %m20: f32,
      %m22: f32,
      %tz: f32,
      %index: index) -> none {
    %out = builtin.unrealized_conversion_cast %output_points : !llvm.ptr to memref<?xf32>
    %in = builtin.unrealized_conversion_cast %input_points : !llvm.ptr to memref<?xf32>
    %idx_i64 = arith.index_cast %index : index to i64
    %idx_i32 = llvm.trunc %idx_i64 overflow<nuw> : i64 to i32
    %idx = arith.index_cast %idx_i32 : i32 to index
    %stride_idx = arith.index_cast %stride : i32 to index
    %base = arith.muli %idx, %stride_idx : index
    %px, %px_done = dataflow.load %in[%base] %ctrl : memref<?xf32>
    %idx_y = arith.index_cast %idx_i32 : i32 to index
    %stride_y = arith.index_cast %stride : i32 to index
    %base_y = arith.muli %idx_y, %stride_y : index
    %off_y = arith.index_cast %y_offset : i32 to index
    %addr_y = arith.addi %base_y, %off_y : index
    %py, %py_done = dataflow.load %in[%addr_y] %ctrl : memref<?xf32>
    %idx_z = arith.index_cast %idx_i32 : i32 to index
    %stride_z = arith.index_cast %stride : i32 to index
    %base_z = arith.muli %idx_z, %stride_z : index
    %off_z = arith.index_cast %z_offset : i32 to index
    %addr_z = arith.addi %base_z, %off_z : index
    %pz, %pz_done = dataflow.load %in[%addr_z] %ctrl : memref<?xf32>
    %x_seed = arith.mulf %m01, %py : f32
    %x_mid = llvm.intr.fmuladd(%m00, %px, %x_seed) : (f32, f32, f32) -> f32
    %x_acc = llvm.intr.fmuladd(%m02, %pz, %x_mid) : (f32, f32, f32) -> f32
    %x = arith.addf %tx, %x_acc : f32
    %x_done = dataflow.store %out[%base] %x %ctrl : memref<?xf32>
    %y_seed = arith.mulf %m11, %py : f32
    %y_mid = llvm.intr.fmuladd(%m10, %px, %y_seed) : (f32, f32, f32) -> f32
    %y_acc = llvm.intr.fmuladd(%m12, %pz, %y_mid) : (f32, f32, f32) -> f32
    %y = arith.addf %ty, %y_acc : f32
    %y_done = dataflow.store %out[%addr_y] %y %ctrl : memref<?xf32>
    %z_seed = arith.mulf %m21, %py : f32
    %z_mid = llvm.intr.fmuladd(%m20, %px, %z_seed) : (f32, f32, f32) -> f32
    %z_acc = llvm.intr.fmuladd(%m22, %pz, %z_mid) : (f32, f32, f32) -> f32
    %z = arith.addf %tz, %z_acc : f32
    %z_done = dataflow.store %out[%addr_z] %z %ctrl : memref<?xf32>
    %done:6 = dataflow.sync %px_done, %py_done, %pz_done, %x_done, %y_done, %z_done
      : (none, none, none, none, none, none) -> (none, none, none, none, none, none)
    dataflow.graph.return %done#0 : none
  }
}
