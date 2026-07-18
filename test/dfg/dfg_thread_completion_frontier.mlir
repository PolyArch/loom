// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-lower %t.dir/valid.mlir -o /dev/null
// RUN: not loom-lower %t.dir/detached.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=DETACHED
// RUN: not loom-lower %t.dir/redundant.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=REDUNDANT
// RUN: not loom-lower %t.dir/missing-independent.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not loom-lower %t.dir/duplicate.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: not loom-lower %t.dir/mismatched-scf-forwarding.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISMATCHED-SCF
// RUN: not loom-lower %t.dir/selective-mux.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=SELECTIVE-MUX
// RUN: not loom-lower %t.dir/superfluous-terminal.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=SUPERFLUOUS
// RUN: not loom-lower %t.dir/incomparable-duplicate-coverage.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=INCOMPARABLE

// DETACHED: thread @detached has graph launch completion not covered by its completion frontier
// REDUNDANT: thread @redundant has a causally redundant completion frontier event
// MISSING: thread @missing_independent has graph launch completion not covered by its completion frontier
// DUPLICATE: thread @duplicate has a duplicate completion frontier event
// MISMATCHED-SCF: thread @mismatched_scf_forwarding has graph launch completion not covered by its completion frontier
// SELECTIVE-MUX: thread @selective_mux has graph launch completion not covered by its completion frontier
// SUPERFLUOUS: thread @superfluous_terminal has a completion frontier event unnecessary for graph launch coverage
// INCOMPARABLE: thread @incomparable_duplicate_coverage has a completion frontier event unnecessary for graph launch coverage

//--- valid.mlir
module {
  dataflow.graph private @work(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }

  dataflow.thread private @direct() ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    dataflow.thread.yield %done : none
  }

  dataflow.thread private @causal_chain() ctrl (%ctrl: none) {
    %first = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    %second = dataflow.graph.launch @work deps(%first) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    dataflow.thread.yield %second : none
  }

  dataflow.thread private @ssa_chain() ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    %terminal = dataflow.sync %done : (none) -> none
    dataflow.thread.yield %terminal : none
  }

  dataflow.thread private @independent() ctrl (%ctrl: none) {
    %first = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    %second = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    dataflow.thread.yield %second, %first : none, none
  }

  dataflow.thread private @forwarded(%condition: i1) ctrl (%ctrl: none) {
    %frontier = scf.if %condition -> (none) {
      %then_done = dataflow.graph.launch @work deps(%ctrl) values()
          stream_inputs() memories() stream_outputs() : (none) -> none
      scf.yield %then_done : none
    } else {
      %else_done = dataflow.graph.launch @work deps(%ctrl) values()
          stream_inputs() memories() stream_outputs() : (none) -> none
      scf.yield %else_done : none
    }
    dataflow.thread.yield %frontier : none
  }

  dataflow.thread private @empty() ctrl (%ctrl: none) {
    dataflow.thread.yield
  }
}

//--- detached.mlir
module {
  dataflow.graph private @work(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @detached() ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    dataflow.thread.yield
  }
}

//--- redundant.mlir
module {
  dataflow.graph private @work(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @redundant() ctrl (%ctrl: none) {
    %first = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    %second = dataflow.graph.launch @work deps(%first) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    dataflow.thread.yield %second, %first : none, none
  }
}

//--- missing-independent.mlir
module {
  dataflow.graph private @work(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @missing_independent() ctrl (%ctrl: none) {
    %first = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    %second = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    dataflow.thread.yield %second : none
  }
}

//--- duplicate.mlir
module {
  dataflow.graph private @work(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @duplicate() ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    dataflow.thread.yield %done, %done : none, none
  }
}

//--- mismatched-scf-forwarding.mlir
module {
  dataflow.graph private @work(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @mismatched_scf_forwarding(%condition: i1)
      ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    %frontier = scf.if %condition -> (none) {
      scf.yield %ctrl : none
    } else {
      scf.yield %done : none
    }
    dataflow.thread.yield %frontier : none
  }
}

//--- selective-mux.mlir
module {
  dataflow.graph private @work(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @selective_mux(%condition: i1) ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    %frontier = dataflow.mux %condition, %ctrl, %done
        : (i1, none, none) -> none
    dataflow.thread.yield %frontier : none
  }
}

//--- superfluous-terminal.mlir
module {
  dataflow.graph private @work(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @superfluous_terminal() ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    %extra = dataflow.sync %ctrl : (none) -> none
    dataflow.thread.yield %done, %extra : none, none
  }
}

//--- incomparable-duplicate-coverage.mlir
module {
  dataflow.graph private @work(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @incomparable_duplicate_coverage()
      ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    %first = dataflow.sync %done : (none) -> none
    %second = dataflow.sync %done : (none) -> none
    dataflow.thread.yield %first, %second : none, none
  }
}
