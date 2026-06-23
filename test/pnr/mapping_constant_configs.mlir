// RUN: loom-pnr-map --dfg-mlir %s --graph constant_two --hardware-mlir %s --hardware constant_adg --workload constant_two --output %t.pass.csv --artifact %t.pass.json
// RUN: FileCheck %s --check-prefix=CSV-PASS < %t.pass.csv
// RUN: FileCheck %s --check-prefix=JSON-PASS < %t.pass.json
// RUN: loom-pnr-map --dfg-mlir %s --graph constant_three --hardware-mlir %s --hardware constant_adg --workload constant_three --output %t.fail.csv --artifact %t.fail.json
// RUN: FileCheck %s --check-prefix=CSV-FAIL < %t.fail.csv
// RUN: loom-pnr-map --dfg-mlir %s --graph shared_constant_five --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload shared_constant_five --output %t.shared-five.csv --artifact %t.shared-five.json
// RUN: FileCheck %s --check-prefix=CSV-SHARED-FIVE < %t.shared-five.csv
// RUN: FileCheck %s --check-prefix=JSON-SHARED-FIVE < %t.shared-five.json

// CSV-PASS: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-PASS-NEXT: constant_two,constant_adg,constant_two__constant_two__constant_adg,1,0,0,0,pass

// JSON-PASS-DAG: "register": "sw_configs.const_hex_value"
// JSON-PASS-DAG: "value": "0x00000002"
// JSON-PASS-DAG: "source": "placement:dataflow.constant#0"

// CSV-FAIL: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-FAIL-NEXT: constant_three,constant_adg,constant_three__constant_three__constant_adg,0,0,0,1,fail,missing hardware resource for software op dataflow.constant

// CSV-SHARED-FIVE: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-SHARED-FIVE-NEXT: shared_constant_five,shared_reduction_adg,shared_constant_five__shared_constant_five__shared_reduction_adg,5,0,0,0,pass

// JSON-SHARED-FIVE-DAG: "software": "dataflow.constant#4"
// JSON-SHARED-FIVE-DAG: "status": "pass"

module {
  dataflow.graph.func private @constant_two(%ctrl: none) -> (none, i32) {
    %value = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    dataflow.graph.return %ctrl, %value : none, i32
  }

  dataflow.graph.func private @constant_three(%ctrl: none) -> (none, i32) {
    %value = dataflow.constant %ctrl {const_value = 3 : i32} : i32
    dataflow.graph.return %ctrl, %value : none, i32
  }

  dataflow.graph.func private @shared_constant_five(%ctrl: none)
      -> (none, i32, i32, i32, i32, i32) {
    %zero0 = dataflow.constant %ctrl {const_value = 0 : i32} : i32
    %one0 = dataflow.constant %ctrl {const_value = 1 : i32} : i32
    %two = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    %zero1 = dataflow.constant %ctrl {const_value = 0 : i32} : i32
    %one1 = dataflow.constant %ctrl {const_value = 1 : i32} : i32
    dataflow.graph.return %ctrl, %zero0, %one0, %two, %zero1, %one1
        : none, i32, i32, i32, i32, i32
  }

  fabric.module @constant_adg(%ctrl: !fabric.bits<0>) {
    fabric.pe [spatial] (%pa = %ctrl : !fabric.bits<0> to !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%input = %pa : !fabric.bits<32> to !fabric.bits<0>)
          -> !fabric.bits<32> {
        %value = fabric.op [@dataflow.constant] (%input)
                 {hw_params = [{const_hex_value = ["0x00000002"]}]}
                 : (!fabric.bits<0>) -> !fabric.bits<32>
        fabric.yield %value : !fabric.bits<32>
      }
    }
    fabric.yield
  }
}
