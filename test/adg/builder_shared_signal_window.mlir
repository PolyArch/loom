// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --shared-signal-window --output %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE
// RUN: loom-pnr-map --dfg-mlir %s --graph signal_window_pressure --hardware-mlir %t.hardware.mlir --hardware shared_signal_window_adg --workload signal_window_pressure --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=MAPPING < %t.dir/mapping.json

// HARDWARE-LABEL: fabric.module @shared_signal_window_adg
// HARDWARE-DAG: load_group_size = 40 : i32
// HARDWARE-DAG: store_group_size = 40 : i32
// HARDWARE-DAG: fabric.op [@dataflow.stream]
// HARDWARE-DAG: fabric.op [@dataflow.carry]
// HARDWARE-DAG: fabric.op [@dataflow.gate]
// HARDWARE-DAG: fabric.op [@dataflow.invariant]
// HARDWARE-DAG: fabric.op [@arith.addf, @arith.subf]
// HARDWARE-DAG: fabric.op [@arith.mulf]
// HARDWARE-DAG: fabric.op [@llvm.fneg]
// HARDWARE-DAG: fabric.op [@dataflow.sync]
// HARDWARE-DAG: fabric.mem [spatial]

// MAPPING-DAG: "workload": "signal_window_pressure"
// MAPPING-DAG: "hardware": "shared_signal_window_adg"
// MAPPING-DAG: "unplaced_records": 0
// MAPPING-DAG: "unrouted_edges": 0
// MAPPING-DAG: "status": "pass"

module {
  dataflow.graph.func private @signal_window_pressure(
      %ctrl: none,
      %lb: i32,
      %ub: i32,
      %step: i32,
      %input: memref<?xf32>,
      %output: memref<?xf32>,
      %scale: f32)
      -> none {
    %zero = dataflow.constant %ctrl {const_value = 0 : index} : index
    %idx, %rwc = dataflow.stream %lb, %ub, %step {cont_cond = ">", step_op = "+="} : i32
    %carried = dataflow.carry %rwc, %lb, %next : i32
    %after_cond, %active_idx = dataflow.gate %rwc, %carried : i32
    %stable_scale = dataflow.invariant %after_cond, %scale : f32
    %value, %load_done = dataflow.load %input[%zero] %ctrl : memref<?xf32>
    %negated = llvm.fneg %value : f32
    %sum = arith.addf %negated, %stable_scale : f32
    %diff = arith.subf %sum, %stable_scale : f32
    %product = arith.mulf %diff, %stable_scale : f32
    %store_done = dataflow.store %output[%zero] %product %ctrl : memref<?xf32>
    %next = arith.addi %active_idx, %step : i32
    %done:2 = dataflow.sync %load_done, %store_done : (none, none) -> (none, none)
    dataflow.graph.return %done#0 : none
  }
}
