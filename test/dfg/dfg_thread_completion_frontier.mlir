// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-lower %t.dir/valid.mlir -o /dev/null
// RUN: not loom-lower %t.dir/detached.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=DETACHED
// RUN: not loom-lower %t.dir/redundant.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=REDUNDANT
// RUN: not loom-lower %t.dir/missing-independent.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not loom-lower %t.dir/duplicate.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=DUPLICATE

// DETACHED: thread @detached has graph launch completion not covered by its completion frontier
// REDUNDANT: thread @redundant has a causally redundant completion frontier event
// MISSING: thread @missing_independent has graph launch completion not covered by its completion frontier
// DUPLICATE: thread @duplicate has a duplicate completion frontier event

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
