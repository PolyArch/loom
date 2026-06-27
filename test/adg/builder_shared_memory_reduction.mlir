// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --shared-memory-reduction --output %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE
// RUN: loom-pnr-map --dfg-mlir %s --graph minmax_pressure --hardware-mlir %t.hardware.mlir --hardware shared_memory_reduction_adg --workload minmax_pressure --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=MAPPING < %t.dir/mapping.json

// HARDWARE-LABEL: fabric.module @shared_memory_reduction_adg
// HARDWARE-DAG: load_group_size = 18 : i32
// HARDWARE-DAG: store_group_size = 9 : i32
// HARDWARE-DAG: fabric.op [@dataflow.constant]
// HARDWARE-DAG: fabric.op [@arith.addi, @arith.subi]
// HARDWARE-DAG: fabric.op [@arith.addf, @arith.subf]
// HARDWARE-DAG: fabric.op [@arith.mulf]
// HARDWARE-DAG: fabric.op [@arith.cmpi, @llvm.icmp]
// HARDWARE-DAG: fabric.op [@arith.cmpf]
// HARDWARE-DAG: fabric.op [@dataflow.sync]
// HARDWARE-DAG: fabric.op [@llvm.intr.smin]
// HARDWARE-DAG: fabric.op [@llvm.intr.smax]
// HARDWARE-DAG: fabric.op [@arith.select]
// HARDWARE-DAG: fabric.mem [spatial]

// MAPPING-DAG: "workload": "minmax_pressure"
// MAPPING-DAG: "hardware": "shared_memory_reduction_adg"
// MAPPING-DAG: "unplaced_records": 0
// MAPPING-DAG: "unrouted_edges": 0
// MAPPING-DAG: "status": "pass"

module {
  dataflow.graph.func private @minmax_pressure(
      %ctrl: none,
      %idx: index,
      %a: memref<?xi8>,
      %b: memref<?xi8>,
      %out: memref<?xi8>,
      %x: i32,
      %y: i32)
      -> none {
    %c0 = dataflow.constant %ctrl {const_value = 0 : index} : index
    %c1 = dataflow.constant %ctrl {const_value = 1 : index} : index
    %c2 = dataflow.constant %ctrl {const_value = 2 : index} : index
    %c3 = dataflow.constant %ctrl {const_value = 3 : index} : index
    %a0, %a0_done = dataflow.load %a[%idx] %ctrl : memref<?xi8>
    %b0, %b0_done = dataflow.load %b[%idx] %ctrl : memref<?xi8>
    %a1, %a1_done = dataflow.load %a[%c1] %ctrl : memref<?xi8>
    %b1, %b1_done = dataflow.load %b[%c1] %ctrl : memref<?xi8>
    %a2, %a2_done = dataflow.load %a[%c2] %ctrl : memref<?xi8>
    %b2, %b2_done = dataflow.load %b[%c2] %ctrl : memref<?xi8>
    %a3, %a3_done = dataflow.load %a[%c3] %ctrl : memref<?xi8>
    %b3, %b3_done = dataflow.load %b[%c3] %ctrl : memref<?xi8>
    %m0 = llvm.intr.smin(%a0, %b0) : (i8, i8) -> i8
    %m1 = llvm.intr.smax(%a1, %b1) : (i8, i8) -> i8
    %p = arith.cmpi sgt, %x, %y : i32
    %s = arith.select %p, %x, %y : i32
    %sum = arith.addi %s, %x : i32
    %m2 = llvm.intr.smin(%a2, %b2) : (i8, i8) -> i8
    %m3 = llvm.intr.smax(%a3, %b3) : (i8, i8) -> i8
    %store0 = dataflow.store %out[%c0] %m0 %ctrl : memref<?xi8>
    %store1 = dataflow.store %out[%c1] %m1 %ctrl : memref<?xi8>
    %store2 = dataflow.store %out[%c2] %m2 %ctrl : memref<?xi8>
    %store3 = dataflow.store %out[%c3] %m3 %ctrl : memref<?xi8>
    dataflow.graph.return %ctrl : none
  }
}
