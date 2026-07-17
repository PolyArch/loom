// RUN: loom-pnr-map --dfg-mlir %s --graph sync9 --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload sync9 --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// CSV-NEXT: sync9,shared_reduction_adg,sync9__sync9__shared_reduction_adg,1,0,0,0,pass,mapped software graph to fabric resources

// JSON-DAG: "status": "pass"
// JSON-DAG: "operation": "dataflow.sync"
// JSON-DAG: "hardware": "shared_reduction_adg::fabric.op#
// JSON-NOT: "resource_kind=fabric.op operation=dataflow.sync

module {
  dataflow.graph.func private @sync9(%ctrl: none, %a: none, %b: none, %c: none,
                                     %d: none, %e: none, %f: none, %g: none,
                                     %h: none)
      -> none
      attributes {input_segments = array<i32: 0, 8, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %done:9 = dataflow.sync %ctrl, %a, %b, %c, %d, %e, %f, %g, %h
        : (none, none, none, none, none, none, none, none, none)
          -> (none, none, none, none, none, none, none, none, none)
    dataflow.graph.return values() streams() memories()
        complete(%done#0 : none)
  }
}
