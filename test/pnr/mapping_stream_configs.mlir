// RUN: loom-pnr-map --dfg-mlir %s --graph stream_add_slt --hardware-mlir %s --hardware stream_add_adg --workload stream_add_slt --output %t.pass.csv --artifact %t.pass.json
// RUN: FileCheck %s --check-prefix=CSV-PASS < %t.pass.csv
// RUN: FileCheck %s --check-prefix=JSON-PASS < %t.pass.json
// RUN: loom-pnr-map --dfg-mlir %s --graph stream_sdiv_sgt --hardware-mlir %s --hardware stream_add_adg --workload stream_sdiv_sgt --output %t.fail.csv --artifact %t.fail.json
// RUN: FileCheck %s --check-prefix=CSV-FAIL < %t.fail.csv

// CSV-PASS: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-PASS-NEXT: stream_add_slt,stream_add_adg,stream_add_slt__stream_add_slt__stream_add_adg,1,0,0,0,pass

// JSON-PASS-DAG: "register": "sw_configs.predicate"
// JSON-PASS-DAG: "value": "slt"
// JSON-PASS-NOT: sw_configs.step_kind

// CSV-FAIL: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-FAIL-NEXT: stream_sdiv_sgt,stream_add_adg,stream_sdiv_sgt__stream_sdiv_sgt__stream_add_adg,0,0,0,1,fail,missing hardware resource for software op dataflow.stream

module {
  dataflow.graph.func private @stream_add_slt(
      %ctrl: none, %init: i32, %limit: i32, %step: i32)
      -> (none, i32, i1) {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while slt : i32
    dataflow.graph.return %ctrl, %iv, %phase : none, i32, i1
  }

  dataflow.graph.func private @stream_sdiv_sgt(
      %ctrl: none, %init: i32, %limit: i32, %step: i32)
      -> (none, i32, i1) {
    %iv, %phase = dataflow.stream %init, %limit, %step
        step sdiv while sgt : i32
    dataflow.graph.return %ctrl, %iv, %phase : none, i32, i1
  }

  fabric.module @stream_add_adg(%init: !fabric.bits<32>,
                                %limit: !fabric.bits<32>,
                                %step: !fabric.bits<32>) {
    fabric.pe [spatial] (%pa = %init : !fabric.bits<32>,
                         %pb = %limit : !fabric.bits<32>,
                         %pc = %step : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%fa = %pa : !fabric.bits<32>,
                %fb = %pb : !fabric.bits<32>,
                %fc = %pc : !fabric.bits<32>) -> !fabric.bits<32> {
        %iv, %phase = fabric.op [@dataflow.stream] (%fa, %fb, %fc)
            {hw_params = [{step_kind = 0 : i32,
                           predicate = [2 : i64, 4 : i64]}]}
            : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
              -> (!fabric.bits<32>, !fabric.bits<1>)
        fabric.yield %iv : !fabric.bits<32>
      }
    }
    fabric.yield
  }
}
