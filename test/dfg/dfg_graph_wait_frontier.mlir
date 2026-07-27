// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-lower %t.dir/valid.mlir -o - | FileCheck %s --check-prefix=VALID
// RUN: not loom-lower %t.dir/uncovered.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNCOVERED
// RUN: not loom-lower %t.dir/nested.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNCOVERED

// A covered retirement frontier survives the canonical lowering path.
// VALID: dataflow.graph.wait %{{.*}}, %{{.*}} : none, none

// The same uncovered frontier is rejected at module scope and under a
// nested builtin.module.
// UNCOVERED: dataflow.graph.wait operand #1 does not cover any graph launch completion event

//--- valid.mlir
module {
  dataflow.graph private @work(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }

  // Direct launch done events and an event whose causal closure covers
  // one form a single unordered all-of retirement frontier.
  dataflow.thread private @mixed domain(#dataflow.thread_domain<dense>)() ctrl (%ctrl: none) {
    %first = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    %second = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    %chained = dataflow.sync %second : (none) -> none
    dataflow.graph.wait %first, %chained : none, none
    dataflow.thread.yield %first, %second : none, none
  }
}

//--- uncovered.mlir
module {
  dataflow.graph private @work(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }

  // A generic none value whose causal closure never reaches a graph
  // launch done event is not a valid wait frontier, even beside a
  // valid one.
  dataflow.thread private @uncovered domain(#dataflow.thread_domain<dense>)() ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    %independent = dataflow.sync %ctrl : (none) -> none
    dataflow.graph.wait %done, %independent : none, none
    dataflow.thread.yield %done : none
  }
}

//--- nested.mlir
module {
  // Coverage is a property of the wait's enclosing thread, not of the
  // module nesting depth at which that thread is defined.
  module {
    dataflow.graph private @work(%start: none) -> ()
        attributes {input_segments = array<i32: 0, 0, 0>,
                    result_segments = array<i32: 0, 0, 0>} {
      dataflow.graph.return %start : none
    }

    dataflow.thread private @uncovered domain(#dataflow.thread_domain<dense>)() ctrl (%ctrl: none) {
      %done = dataflow.graph.launch @work deps(%ctrl) values()
          stream_inputs() memories() stream_outputs() : (none) -> none
      %independent = dataflow.sync %ctrl : (none) -> none
      dataflow.graph.wait %done, %independent : none, none
      dataflow.thread.yield %done : none
    }
  }
}
