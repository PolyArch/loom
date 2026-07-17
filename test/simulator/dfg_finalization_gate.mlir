// RUN: not loom-dfg-sim %s --graph residual_for --output %t.residual.json 2>&1 | FileCheck %s --check-prefix=RESIDUAL
// RUN: not loom-dfg-sim %s --graph raw_start_work --output %t.start.json 2>&1 | FileCheck %s --check-prefix=START
// RUN: not loom-dfg-sim %s --graph uncovered_value --output %t.value.json 2>&1 | FileCheck %s --check-prefix=VALUE
// RUN: loom-dfg-sim %s --graph detached_actor --arg 0=1 --output %t.detached.json
// RUN: FileCheck %s --check-prefix=DETACHED < %t.detached.json

// RESIDUAL: finalized graph contains residual structured operation 'scf.for'
// START: nontrivial graph uses raw start as a retirement completion witness
// VALUE: retirement frontier does not causally cover value output #0
// DETACHED-DAG: "status": "pass"
// DETACHED-NOT: "arith.addi"
// DETACHED-NOT: fired after graph retirement

module {
  dataflow.graph.func private @residual_for(
      %start: none, %lb: index, %ub: index, %step: index) -> none
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    scf.for %i = %lb to %ub step %step {
    }
    dataflow.graph.return %start : none
  }

  dataflow.graph.func private @raw_start_work(%start: none) -> (none, i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %start {const_value = 7 : i32} : i32
    dataflow.graph.return %start, %value : none, i32
  }

  dataflow.graph.func private @uncovered_value(%start: none) -> (none, i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %start {const_value = 7 : i32} : i32
    %complete = dataflow.sync %start : (none) -> none
    dataflow.graph.return %complete, %value : none, i32
  }

  dataflow.graph.func private @detached_actor(%start: none, %input: i32)
      -> (none, i32)
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %published:2 = dataflow.sync %start, %input
        : (none, i32) -> (none, i32)
    %first = arith.addi %input, %input : i32
    %detached = arith.addi %first, %input : i32
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}
