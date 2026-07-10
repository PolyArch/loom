// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --shared-quantized-window --output %t.dir/hardware.mlir
// RUN: loom-adg-builder-test --shared-quantized-window --output %t.dir/hardware.second.mlir
// RUN: cmp %t.dir/hardware.mlir %t.dir/hardware.second.mlir
// RUN: loom %t.dir/hardware.mlir | FileCheck %s --check-prefix=HARDWARE
// RUN: loom-pnr-map --dfg-mlir %s --graph quantized_window_pressure --hardware-mlir %t.dir/hardware.mlir --hardware shared_quantized_window_adg --workload quantized_window_pressure --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=MAPPING < %t.dir/mapping.json

// HARDWARE-LABEL: fabric.module @shared_quantized_window_adg
// HARDWARE-DAG: load_group_size = 18 : i32
// HARDWARE-DAG: store_group_size = 9 : i32
// HARDWARE-DAG: fabric.op [@dataflow.stream]
// HARDWARE-DAG: fabric.op [@dataflow.carry]
// HARDWARE-DAG: fabric.op [@dataflow.invariant]
// HARDWARE-DAG: fabric.op [@dataflow.constant]
// HARDWARE-DAG: 0x0000ffef
// HARDWARE-DAG: 0x00000018
// HARDWARE-DAG: 0x30000000
// HARDWARE-DAG: 0xffff0000
// HARDWARE-DAG: fabric.op [@arith.addi, @arith.subi]
// HARDWARE-DAG: fabric.op [@arith.muli]
// HARDWARE-DAG: fabric.op [@arith.divsi]
// HARDWARE-DAG: fabric.op [@arith.remsi]
// HARDWARE-DAG: fabric.op [@arith.cmpi, @llvm.icmp]
// HARDWARE-DAG: fabric.op [@arith.shli, @arith.shrsi, @arith.shrui]
// HARDWARE-DAG: fabric.op [@llvm.intr.fshl]
// HARDWARE-DAG: fabric.op [@llvm.arm.pkhbt]
// HARDWARE-DAG: fabric.op [@llvm.arm.pkhtb]
// HARDWARE-DAG: fabric.op [@llvm.arm.sadd16]
// HARDWARE-DAG: fabric.op [@llvm.arm.sxtab16]
// HARDWARE-DAG: fabric.op [@llvm.arm.sxtb16]
// HARDWARE-DAG: fabric.op [@dataflow.mux]
// HARDWARE-DAG: fabric.op [@llvm.intr.ctlz]
// HARDWARE-DAG: fabric.op [@llvm.intr.umax]
// HARDWARE-DAG: fabric.op [@llvm.intr.smin]
// HARDWARE-DAG: fabric.op [@llvm.intr.smax]
// HARDWARE-DAG: fabric.op [@arith.select]
// HARDWARE-DAG: fabric.op [@llvm.sext]
// HARDWARE-DAG: fabric.mem [spatial]

// MAPPING-DAG: "workload": "quantized_window_pressure"
// MAPPING-DAG: "hardware": "shared_quantized_window_adg"
// MAPPING-DAG: "operation": "dataflow.stream"
// MAPPING-DAG: "operation": "dataflow.carry"
// MAPPING-DAG: "operation": "dataflow.invariant"
// MAPPING-DAG: "unplaced_records": 0
// MAPPING-DAG: "unrouted_edges": 0
// MAPPING-DAG: "status": "pass"
// MAPPING-NOT: "resource_pressure"

module {
  dataflow.graph.func private @quantized_window_pressure(
      %ctrl: none,
      %idx: index,
      %in: memref<?xi8>,
      %out: memref<?xi8>,
      %out32: memref<?xi32>,
      %x: i32,
      %y: i32)
      -> none {
    %c0 = dataflow.constant %ctrl {const_value = 0 : index} : index
    %c1 = dataflow.constant %ctrl {const_value = 1 : index} : index
    %c2 = dataflow.constant %ctrl {const_value = 2 : index} : index
    %c3 = dataflow.constant %ctrl {const_value = 3 : index} : index
    %i0 = dataflow.constant %ctrl {const_value = 0 : i32} : i32
    %i1 = dataflow.constant %ctrl {const_value = 1 : i32} : i32
    %i3 = dataflow.constant %ctrl {const_value = 3 : i32} : i32
    %one = dataflow.constant %ctrl {const_value = 1 : i32} : i32
    %loop_idx, %rwc = dataflow.stream %i0, %i3, %i1 {cont_cond = "<", step_op = "+="} : i32
    %stable_x = dataflow.invariant %rwc, %x : i32
    %carried = dataflow.carry %rwc, %stable_x, %next : i32
    %next = arith.addi %carried, %one : i32
    %loop_adjusted = arith.addi %loop_idx, %next : i32
    %a0, %a0_done = dataflow.load %in[%idx] %ctrl : memref<?xi8>
    %a1, %a1_done = dataflow.load %in[%c1] %ctrl : memref<?xi8>
    %a2, %a2_done = dataflow.load %in[%c2] %ctrl : memref<?xi8>
    %a3, %a3_done = dataflow.load %in[%c3] %ctrl : memref<?xi8>
    %m0 = llvm.intr.smax(%a0, %a1) : (i8, i8) -> i8
    %m1 = llvm.intr.smax(%a2, %a3) : (i8, i8) -> i8
    %p0 = arith.cmpi sgt, %x, %y : i32
    %p1 = arith.cmpi slt, %x, %y : i32
    %s0 = arith.shli %x, %y : i32
    %s1 = arith.shrsi %s0, %y : i32
    %s2 = arith.shrui %s1, %y : i32
    %d0 = arith.subi %s2, %x : i32
    %u0 = arith.addi %d0, %loop_adjusted : i32
    %q0 = arith.divsi %u0, %one : i32
    %q1 = arith.divsi %x, %one : i32
    %q2 = arith.divsi %y, %one : i32
    %q3 = arith.divsi %s2, %one : i32
    %sel0 = arith.select %p0, %u0, %x : i32
    %sel1 = arith.select %p1, %sel0, %y : i32
    %mux0 = dataflow.mux %p0, %q0, %q1 : (i1, i32, i32) -> i32
    %mux1 = dataflow.mux %p1, %q2, %q3 : (i1, i32, i32) -> i32
    %mux2 = dataflow.mux %p0, %mux0, %mux1 : (i1, i32, i32) -> i32
    %mux3 = dataflow.mux %p1, %mux2, %sel1 : (i1, i32, i32) -> i32
    %store0 = dataflow.store %out[%c0] %m0 %ctrl : memref<?xi8>
    %store1 = dataflow.store %out[%c1] %m1 %ctrl : memref<?xi8>
    %store2 = dataflow.store %out32[%c2] %mux3 %ctrl : memref<?xi32>
    dataflow.graph.return %ctrl : none
  }
}
