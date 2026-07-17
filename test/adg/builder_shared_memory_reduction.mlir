// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --shared-memory-reduction --output %t.dir/hardware.mlir
// RUN: loom-adg-builder-test --shared-memory-reduction --output %t.dir/hardware.second.mlir
// RUN: cmp %t.dir/hardware.mlir %t.dir/hardware.second.mlir
// RUN: FileCheck %s --check-prefix=BUILDER < %t.dir/hardware.mlir
// RUN: loom %t.dir/hardware.mlir | FileCheck %s --check-prefix=HARDWARE
// RUN: loom-pnr-map --dfg-mlir %s --graph minmax_pressure --hardware-mlir %t.dir/hardware.mlir --hardware shared_memory_reduction_adg --workload minmax_pressure --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=MAPPING < %t.dir/mapping.json
// RUN: mkdir -p %t.dir/bitonic_stage
// RUN: %loom-cc -emit-llvm -O1 -S %S/../app/bitonic_stage/main_func.c -o %t.dir/bitonic_stage/main_func.ll
// RUN: %loom-raise %t.dir/bitonic_stage/main_func.ll -o %t.dir/bitonic_stage/main_func.scf.mlir
// RUN: %loom-lower %t.dir/bitonic_stage/main_func.scf.mlir -o %t.dir/bitonic_stage/main_func.dfg.mlir
// RUN: FileCheck %s --check-prefix=BITONIC-DFG < %t.dir/bitonic_stage/main_func.dfg.mlir
// RUN: loom-pnr-map --dfg-mlir %t.dir/bitonic_stage/main_func.dfg.mlir --graph g_bitonic_stage_0 --hardware-mlir %t.dir/hardware.mlir --hardware shared_memory_reduction_adg --workload bitonic_stage --output %t.dir/bitonic_stage.mapping.csv --artifact %t.dir/bitonic_stage.mapping.json
// RUN: FileCheck %s --check-prefix=BITONIC < %t.dir/bitonic_stage.mapping.json

// BITONIC-DFG-LABEL: dataflow.graph.func private @g_bitonic_stage_0
// BITONIC-DFG: builtin.unrealized_conversion_cast %arg1 : !llvm.ptr to memref<?xf32>
// BITONIC-DFG-NOT: dataflow.demux {{.*}} : (i1, !llvm.ptr)
// BITONIC-DFG-NOT: dataflow.invariant {{.*}} : !llvm.ptr
// BITONIC-DFG: dataflow.stream
// BITONIC-DFG: arith.index_cast {{.*}} : i64 to index
// BITONIC-DFG-NOT: dataflow.demux {{.*}} : (i1, !llvm.ptr)
// BITONIC-DFG-NOT: dataflow.invariant {{.*}} : !llvm.ptr
// BITONIC-DFG: dataflow.graph.return

// HARDWARE-LABEL: fabric.module @shared_memory_reduction_adg
// HARDWARE-DAG: load_group_size = 18 : i32
// HARDWARE-DAG: store_group_size = 9 : i32
// HARDWARE-DAG: fabric.op [@dataflow.constant]
// HARDWARE-DAG: const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0x0000001e", "0x0000003f", "0xffffffff"]
// HARDWARE-DAG: const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x0000001f", "0x40000000"]
// HARDWARE-DAG: fabric.op [@arith.addi, @arith.subi]
// HARDWARE-DAG: fabric.op [@arith.muli]
// HARDWARE-DAG: fabric.op [@arith.divui, @arith.remui]
// HARDWARE-DAG: fabric.op [@arith.addf, @arith.subf]
// HARDWARE-DAG: fabric.op [@arith.mulf]
// HARDWARE-DAG: fabric.op [@llvm.intr.fmuladd]
// HARDWARE-DAG: fabric.op [@arith.cmpi, @llvm.icmp]
// HARDWARE-DAG: fabric.op [@arith.cmpf]
// HARDWARE-DAG: fabric.op [@dataflow.sync]
// HARDWARE-DAG: fabric.op [@llvm.intr.umin]
// HARDWARE-DAG: fabric.op [@llvm.intr.smin]
// HARDWARE-DAG: fabric.op [@llvm.intr.smax]
// HARDWARE-DAG: fabric.op [@arith.select]
// HARDWARE-DAG: fabric.op [@llvm.select]
// HARDWARE-DAG: fabric.op [@dataflow.demux]
// HARDWARE-DAG: fabric.op [@llvm.trunc]
// HARDWARE-DAG: fabric.op [@llvm.zext]
// HARDWARE-DAG: fabric.op [@dataflow.stream]
// HARDWARE-DAG: fabric.op [@dataflow.carry]
// HARDWARE-DAG: fabric.op [@dataflow.invariant]
// HARDWARE-DAG: fabric.op [@dataflow.gate]
// HARDWARE-DAG: fabric.op [@arith.index_cast]
// HARDWARE-DAG: fabric.mem [spatial]

// BUILDER-LABEL: fabric.module @shared_memory_reduction_adg
// BUILDER-DAG: %wide_const0 =
// BUILDER-DAG: %wide_const1 =
// BUILDER-DAG: %udiv0 =
// BUILDER-DAG: %wide_mul0 =
// BUILDER-DAG: %wide_udiv0 =
// BUILDER-DAG: %wide_shift0 =
// BUILDER-DAG: %wide_cmp0 =
// BUILDER-DAG: %wide_cmp0_pred = fabric.fifo %wide_cmp0
// BUILDER-DAG: %wide_mux0 =
// BUILDER-DAG: %demux0_false, %demux0_true =
// BUILDER-DAG: %control_demux0_false_wide, %control_demux0_true_wide =
// BUILDER-DAG: %control_demux0_false = fabric.fifo %control_demux0_false_wide
// BUILDER-DAG: %wide_route_bridge0 = fabric.fifo %wide_route_bridge0_input

// MAPPING-DAG: "workload": "minmax_pressure"
// MAPPING-DAG: "hardware": "shared_memory_reduction_adg"
// MAPPING-DAG: "unplaced_records": 0
// MAPPING-DAG: "unrouted_edges": 0
// MAPPING-DAG: "status": "pass"

// BITONIC-DAG: "workload": "bitonic_stage"
// BITONIC-DAG: "hardware": "shared_memory_reduction_adg"
// BITONIC-DAG: "unplaced_records": 0
// BITONIC-DAG: "unrouted_edges": 0
// BITONIC-DAG: "software": "arith.cmpi#0"
// BITONIC-DAG: "software": "dataflow.store#0"
// BITONIC-DAG: "edge_ref": "arith.cmpi#0.result0->arith.select#0.operand0"
// BITONIC-DAG: "edge_ref": "dataflow.constant#1.result0->arith.shli#0.operand1"
// BITONIC-DAG: "edge_ref": "dataflow.load#1.result0->arith.cmpf#1.operand1"
// BITONIC-DAG: "status": "pass"

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
