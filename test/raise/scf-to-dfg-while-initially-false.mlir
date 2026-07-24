// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph while_initially_false \
// RUN:   --max-event-steps=64 --output %t.json
// RUN: FileCheck %s --check-prefix=SIM < %t.json

// SIM-DAG: "graph": "while_initially_false"
// SIM-DAG: "status": "pass"

dataflow.graph private @while_initially_false(%start: none) -> ()
    attributes {input_segments = array<i32: 0, 0, 0>,
                result_segments = array<i32: 0, 0, 0>} {
  %false = arith.constant false
  scf.while : () -> () {
    scf.condition(%false)
  } do {
    scf.yield
  }
  dataflow.graph.return %start : none
}
