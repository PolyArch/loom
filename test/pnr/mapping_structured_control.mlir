// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: not loom-pnr-map --dfg-mlir %t.dir/residual.mlir --graph residual_for --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload residual_for --output %t.residual.csv --artifact %t.residual.json 2>&1 | FileCheck %s --check-prefix=RESIDUAL
// RUN: not loom-pnr-map --dfg-mlir %t.dir/start.mlir --graph raw_start_work --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload raw_start_work --output %t.start.csv --artifact %t.start.json 2>&1 | FileCheck %s --check-prefix=START

// RESIDUAL: finalized graph contains residual structured operation 'scf.for'
// START: nontrivial graph uses raw start as a retirement completion witness

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

//--- start.mlir
module {
  dataflow.graph private @raw_start_work(%start: none) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.constant %start {const_value = 7 : i32} : i32
    dataflow.graph.return %start, %value : none, i32
  }
}
