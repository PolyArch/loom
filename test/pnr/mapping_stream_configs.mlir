// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph stream_add_slt --hardware-mlir %s --hardware stream_add_i32_adg --workload stream_add_slt --output %t.i32-pass.csv --artifact %t.i32-pass.json
// RUN: FileCheck %s --check-prefix=CSV-I32-PASS < %t.i32-pass.csv
// RUN: FileCheck %s --check-prefix=JSON-I32-PASS < %t.i32-pass.json
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph stream_add_slt_i64 --hardware-mlir %s --hardware stream_add_i64_adg --workload stream_add_slt_i64 --output %t.i64-pass.csv --artifact %t.i64-pass.json
// RUN: FileCheck %s --check-prefix=CSV-I64-PASS < %t.i64-pass.csv
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph stream_add_slt_i64 --hardware-mlir %s --hardware stream_add_i32_adg --workload stream_i64_on_i32 --output %t.i64-on-i32.csv --artifact %t.i64-on-i32.json
// RUN: FileCheck %s --check-prefix=CSV-I64-ON-I32 < %t.i64-on-i32.csv
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph stream_add_slt --hardware-mlir %s --hardware stream_add_i64_adg --workload stream_i32_on_i64 --output %t.i32-on-i64.csv --artifact %t.i32-on-i64.json
// RUN: FileCheck %s --check-prefix=CSV-I32-ON-I64 < %t.i32-on-i64.csv
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph stream_sdiv_sgt --hardware-mlir %s --hardware stream_add_i32_adg --workload stream_sdiv_sgt --output %t.config-fail.csv --artifact %t.config-fail.json
// RUN: FileCheck %s --check-prefix=CSV-CONFIG-FAIL < %t.config-fail.csv

// CSV-I32-PASS: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-I32-PASS-NEXT: stream_add_slt,stream_add_i32_adg,stream_add_slt__stream_add_slt__stream_add_i32_adg,2,1,0,0,pass

// JSON-I32-PASS-DAG: "register": "sw_configs.predicate"
// JSON-I32-PASS-DAG: "value": "slt"
// JSON-I32-PASS-NOT: sw_configs.step_kind

// CSV-I64-PASS: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-I64-PASS-NEXT: stream_add_slt_i64,stream_add_i64_adg,stream_add_slt_i64__stream_add_slt_i64__stream_add_i64_adg,2,1,0,0,pass

// CSV-I64-ON-I32: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-I64-ON-I32-NEXT: stream_i64_on_i32,stream_add_i32_adg,stream_i64_on_i32__stream_add_slt_i64__stream_add_i32_adg,1,0,0,1,fail,missing hardware resource for software op dataflow.stream

// CSV-I32-ON-I64: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-I32-ON-I64-NEXT: stream_i32_on_i64,stream_add_i64_adg,stream_i32_on_i64__stream_add_slt__stream_add_i64_adg,1,0,0,1,fail,missing hardware resource for software op dataflow.stream

// CSV-CONFIG-FAIL: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-CONFIG-FAIL-NEXT: stream_sdiv_sgt,stream_add_i32_adg,stream_sdiv_sgt__stream_sdiv_sgt__stream_add_i32_adg,1,0,0,1,fail,missing hardware resource for software op dataflow.stream

module {
  dataflow.graph.func private @stream_add_slt(
      %ctrl: none, %init: i32, %limit: i32, %step: i32, %unit: none)
      -> (none, i32, i1)
      attributes {input_segments = array<i32: 3, 1, 0>,
                  result_segments = array<i32: 0, 2, 0>} {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while slt : i32
    %complete:2 = dataflow.demux %phase, %unit
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%iv, %phase : i32, i1) memories()
        complete(%complete#0 : none)
  }

  dataflow.graph.func private @stream_sdiv_sgt(
      %ctrl: none, %init: i32, %limit: i32, %step: i32, %unit: none)
      -> (none, i32, i1)
      attributes {input_segments = array<i32: 3, 1, 0>,
                  result_segments = array<i32: 0, 2, 0>} {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step sdiv while sgt : i32
    %complete:2 = dataflow.demux %phase, %unit
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%iv, %phase : i32, i1) memories()
        complete(%complete#0 : none)
  }

  dataflow.graph.func private @stream_add_slt_i64(
      %ctrl: none, %init: i64, %limit: i64, %step: i64, %unit: none)
      -> (none, i64, i1)
      attributes {input_segments = array<i32: 3, 1, 0>,
                  result_segments = array<i32: 0, 2, 0>} {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while slt : i64
    %complete:2 = dataflow.demux %phase, %unit
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%iv, %phase : i64, i1) memories()
        complete(%complete#0 : none)
  }

  fabric.module @stream_add_i32_adg(%ctrl: !fabric.bits<0>,
                                    %init: !fabric.bits<32>,
                                    %limit: !fabric.bits<32>,
                                    %step: !fabric.bits<32>) {
    fabric.pe [spatial] (%pa = %init : !fabric.bits<32>,
                         %pb = %limit : !fabric.bits<32>,
                         %pc = %step : !fabric.bits<32>,
                         %pd = %ctrl : !fabric.bits<0> to !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%fa = %pa : !fabric.bits<32>,
                %fb = %pb : !fabric.bits<32>,
                %fc = %pc : !fabric.bits<32>,
                %unit = %pd : !fabric.bits<32> to !fabric.bits<0>)
          -> !fabric.bits<32> {
        %iv, %phase = fabric.op [@dataflow.stream] (%fa, %fb, %fc)
            {hw_params = [{step_kind = 0 : i32,
                           predicate = [2 : i64, 4 : i64]}]}
            : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
              -> (!fabric.bits<32>, !fabric.bits<1>)
        %closed, %active = fabric.op [@dataflow.demux] (%phase, %unit)
            : (!fabric.bits<1>, !fabric.bits<0>)
              -> (!fabric.bits<0>, !fabric.bits<0>)
        fabric.yield %iv : !fabric.bits<32>
      }
    }
    fabric.yield
  }

  fabric.module @stream_add_i64_adg(%ctrl: !fabric.bits<0>,
                                    %init: !fabric.bits<64>,
                                    %limit: !fabric.bits<64>,
                                    %step: !fabric.bits<64>) {
    fabric.pe [spatial] (%pa = %init : !fabric.bits<64>,
                         %pb = %limit : !fabric.bits<64>,
                         %pc = %step : !fabric.bits<64>,
                         %pd = %ctrl : !fabric.bits<0> to !fabric.bits<64>)
        -> !fabric.bits<64> {
      fabric.fu(%fa = %pa : !fabric.bits<64>,
                %fb = %pb : !fabric.bits<64>,
                %fc = %pc : !fabric.bits<64>,
                %unit = %pd : !fabric.bits<64> to !fabric.bits<0>)
          -> !fabric.bits<64> {
        %iv, %phase = fabric.op [@dataflow.stream] (%fa, %fb, %fc)
            {hw_params = [{step_kind = 0 : i32,
                           predicate = [2 : i64, 4 : i64]}]}
            : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
              -> (!fabric.bits<64>, !fabric.bits<1>)
        %closed, %active = fabric.op [@dataflow.demux] (%phase, %unit)
            : (!fabric.bits<1>, !fabric.bits<0>)
              -> (!fabric.bits<0>, !fabric.bits<0>)
        fabric.yield %iv : !fabric.bits<64>
      }
    }
    fabric.yield
  }
}
