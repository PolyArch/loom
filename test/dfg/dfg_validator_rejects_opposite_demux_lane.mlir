// RUN: not loom-dfg-sim %s --graph opposite_demux_completion --arg 0=true --output %t.sim.json 2>&1 | FileCheck %s --check-prefix=SIM
// RUN: not loom-pnr-map --dfg-mlir %s --graph opposite_demux_completion --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload opposite_demux_completion --output %t.mapping.csv --artifact %t.mapping.json 2>&1 | FileCheck %s --check-prefix=PNR

// SIM: retirement frontier does not causally cover stream output #0
// PNR: retirement frontier does not causally cover stream output #0

// The two demux outputs are selected alternatives. Completing the false lane
// cannot retire the stream exported on the true lane merely because both
// values were produced by one operation.
module {
  dataflow.graph.func private @opposite_demux_completion(
      %start: none, %select: i1) -> (none, none)
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 1, 0>} {
    %lanes:2 = dataflow.demux %select, %start : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%lanes#1 : none) memories()
        complete(%lanes#0 : none)
  }
}
