// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: not loom-dfg-sim %t.dir/residual.mlir --graph residual_for --output %t.residual.json 2>&1 | FileCheck %s --check-prefix=RESIDUAL
// RUN: not loom-dfg-sim %t.dir/cf.mlir --graph residual_assert --arg 0=true --output %t.cf.json 2>&1 | FileCheck %s --check-prefix=CF
// RUN: not loom-dfg-sim %t.dir/start.mlir --graph raw_start_work --output %t.start.json 2>&1 | FileCheck %s --check-prefix=START
// RUN: not loom-dfg-sim %t.dir/value.mlir --graph uncovered_value --output %t.value.json 2>&1 | FileCheck %s --check-prefix=VALUE
// RUN: loom-dfg-sim %t.dir/detached.mlir --graph detached_actor --arg 0=1 --output %t.detached.json
// RUN: FileCheck %s --check-prefix=DETACHED < %t.detached.json
// RUN: loom-dfg-sim %t.dir/detached-failure.mlir --graph detached_failure --arg 0=7 --output %t.detached-failure.json
// RUN: FileCheck %s --check-prefix=DETACHED-FAILURE < %t.detached-failure.json

// RESIDUAL: finalized graph contains residual structured operation 'scf.for'
// CF: finalized graph contains residual structured operation 'cf.assert'
// START: nontrivial graph uses raw start as a retirement completion witness
// VALUE: retirement frontier does not causally cover value output #0
// DETACHED-DAG: "status": "invalid"
// DETACHED-DAG: actor 'arith.addi' fired after graph retirement
// DETACHED-FAILURE-DAG: "status": "invalid"
// DETACHED-FAILURE-DAG: arith.divsi division by zero is undefined
// DETACHED-FAILURE-DAG: actor 'arith.divsi' failed after graph retirement

//--- residual.mlir
module {
  dataflow.graph private @residual_for(
      %start: none, %lb: index, %ub: index, %step: index) -> ()
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    scf.for %i = %lb to %ub step %step {
    }
    dataflow.graph.return %start : none
  }
}

//--- cf.mlir
module {
  dataflow.graph private @residual_assert(%start: none, %condition: i1) -> ()
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    cf.assert %condition, "residual assertion"
    dataflow.graph.return %start : none
  }
}

//--- detached-failure.mlir
module {
  dataflow.graph private @detached_failure(%start: none, %input: i32)
      -> (i32)
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i32} : i32
    %published:2 = dataflow.sync %start, %input
        : (none, i32) -> (none, i32)
    %quotient = arith.divsi %input, %zero : i32
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}

//--- start.mlir
module {
  dataflow.graph private @raw_start_work(%start: none) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %start {const_value = 7 : i32} : i32
    dataflow.graph.return %start, %value : none, i32
  }
}

//--- value.mlir
module {
  dataflow.graph private @uncovered_value(%start: none) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %start {const_value = 7 : i32} : i32
    %complete = dataflow.sync %start : (none) -> none
    dataflow.graph.return %complete, %value : none, i32
  }
}

//--- detached.mlir
module {
  dataflow.graph private @detached_actor(%start: none, %input: i32)
      -> (i32)
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %published:2 = dataflow.sync %start, %input
        : (none, i32) -> (none, i32)
    %first = arith.addi %input, %input : i32
    %detached = arith.addi %first, %input : i32
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}
