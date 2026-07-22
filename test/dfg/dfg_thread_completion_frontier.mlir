// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-lower %t.dir/valid.mlir -o /dev/null
// RUN: loom-dfg-sim %t.dir/for-valid.mlir --graph work --output %t.for.json
// RUN: loom-dfg-sim %t.dir/while-valid.mlir --graph work --output %t.while.json
// RUN: not loom-lower %t.dir/detached.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=DETACHED
// RUN: not loom-lower %t.dir/redundant.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=REDUNDANT
// RUN: not loom-lower %t.dir/missing-independent.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not loom-lower %t.dir/duplicate.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: not loom-lower %t.dir/mismatched-scf-forwarding.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISMATCHED-SCF
// RUN: not loom-lower %t.dir/selective-mux.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=SELECTIVE-MUX
// RUN: not loom-lower %t.dir/superfluous-terminal.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=SUPERFLUOUS
// RUN: not loom-lower %t.dir/incomparable-duplicate-coverage.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=INCOMPARABLE
// RUN: not loom-lower %t.dir/repetitive-latest-only.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPETITIVE
// RUN: not loom-lower %t.dir/nested-module.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=NESTED-MODULE
// RUN: not loom-dfg-sim %t.dir/for-latest-only.mlir --graph work --output %t.for-invalid.json 2>&1 | FileCheck %s --check-prefix=FOR-REPETITIVE
// RUN: not loom-dfg-sim %t.dir/while-reordered-latest-only.mlir --graph work --output %t.while-invalid.json 2>&1 | FileCheck %s --check-prefix=WHILE-REORDERED

// DETACHED: thread @detached has graph launch completion not covered by its completion frontier
// REDUNDANT: thread @redundant has a causally redundant completion frontier event
// MISSING: thread @missing_independent has graph launch completion not covered by its completion frontier
// DUPLICATE: thread @duplicate has a duplicate completion frontier event
// MISMATCHED-SCF: thread @mismatched_scf_forwarding has graph launch completion not covered by its completion frontier
// SELECTIVE-MUX: thread @selective_mux has graph launch completion not covered by its completion frontier
// SUPERFLUOUS: thread @superfluous_terminal has a completion frontier event unnecessary for graph launch coverage
// INCOMPARABLE: thread @incomparable_duplicate_coverage has a completion frontier event unnecessary for graph launch coverage
// REPETITIVE: thread @repetitive_latest_only has graph launch completion not covered by its completion frontier
// NESTED-MODULE: thread @nested_module has graph launch completion not covered by its completion frontier
// FOR-REPETITIVE: thread @for_latest_only has graph launch completion not covered by its completion frontier
// WHILE-REORDERED: thread @while_reordered_latest_only has graph launch completion not covered by its completion frontier

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

  dataflow.thread private @one_sided(%condition: i1) ctrl (%ctrl: none) {
    %frontier = scf.if %condition -> (none) {
      %done = dataflow.graph.launch @work deps(%ctrl) values()
          stream_inputs() memories() stream_outputs() : (none) -> none
      scf.yield %done : none
    } else {
      scf.yield %ctrl : none
    }
    dataflow.thread.yield %frontier : none
  }

  dataflow.thread private @selector_matched(%condition: i1) ctrl (%ctrl: none) {
    %starts:2 = dataflow.demux %condition, %ctrl
        : (i1, none) -> (none, none)
    %done = dataflow.graph.launch @work deps(%starts#1) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    %frontier = dataflow.mux %condition, %starts#0, %done
        : (i1, none, none) -> none
    dataflow.thread.yield %frontier : none
  }

  dataflow.thread private @loop_chained(%limit: index) ctrl (%ctrl: none) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %result:2 = scf.while (%i = %zero, %latest = %ctrl)
        : (index, none) -> (index, none) {
      %more = arith.cmpi ult, %i, %limit : index
      scf.condition(%more) %i, %latest : index, none
    } do {
    ^bb0(%i: index, %latest: none):
      %done = dataflow.graph.launch @work deps(%latest) values()
          stream_inputs() memories() stream_outputs() : (none) -> none
      %next = arith.addi %i, %one : index
      scf.yield %next, %done : index, none
    }
    dataflow.thread.yield %result#1 : none
  }

  dataflow.thread private @loop_zero_trip(%limit: index) ctrl (%ctrl: none) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %done = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    %result:2 = scf.while (%i = %zero, %carry = %done)
        : (index, none) -> (index, none) {
      %more = arith.cmpi ult, %i, %limit : index
      scf.condition(%more) %i, %carry : index, none
    } do {
    ^bb0(%i: index, %carry: none):
      %next = arith.addi %i, %one : index
      scf.yield %next, %carry : index, none
    }
    dataflow.thread.yield %result#1 : none
  }

  dataflow.thread private @shared_dag(%selector: index) ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    %level0 = dataflow.mux %selector, %done, %done, %done, %done,
        %done, %done, %done, %done
        : (index, none, none, none, none, none, none, none, none) -> none
    %level1 = dataflow.mux %selector, %level0, %level0, %level0, %level0,
        %level0, %level0, %level0, %level0
        : (index, none, none, none, none, none, none, none, none) -> none
    %level2 = dataflow.mux %selector, %level1, %level1, %level1, %level1,
        %level1, %level1, %level1, %level1
        : (index, none, none, none, none, none, none, none, none) -> none
    %level3 = dataflow.mux %selector, %level2, %level2, %level2, %level2,
        %level2, %level2, %level2, %level2
        : (index, none, none, none, none, none, none, none, none) -> none
    %level4 = dataflow.mux %selector, %level3, %level3, %level3, %level3,
        %level3, %level3, %level3, %level3
        : (index, none, none, none, none, none, none, none, none) -> none
    %level5 = dataflow.mux %selector, %level4, %level4, %level4, %level4,
        %level4, %level4, %level4, %level4
        : (index, none, none, none, none, none, none, none, none) -> none
    %level6 = dataflow.mux %selector, %level5, %level5, %level5, %level5,
        %level5, %level5, %level5, %level5
        : (index, none, none, none, none, none, none, none, none) -> none
    %level7 = dataflow.mux %selector, %level6, %level6, %level6, %level6,
        %level6, %level6, %level6, %level6
        : (index, none, none, none, none, none, none, none, none) -> none
    %level8 = dataflow.mux %selector, %level7, %level7, %level7, %level7,
        %level7, %level7, %level7, %level7
        : (index, none, none, none, none, none, none, none, none) -> none
    %level9 = dataflow.mux %selector, %level8, %level8, %level8, %level8,
        %level8, %level8, %level8, %level8
        : (index, none, none, none, none, none, none, none, none) -> none
    dataflow.thread.yield %level9 : none
  }

  dataflow.thread private @empty() ctrl (%ctrl: none) {
    dataflow.thread.yield
  }
}

//--- while-valid.mlir
module {
  dataflow.graph private @work(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @while_reordered_chained(%limit: index)
      ctrl (%ctrl: none) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %result:2 = scf.while (%carry = %ctrl, %i = %zero)
        : (none, index) -> (index, none) {
      %more = arith.cmpi ult, %i, %limit : index
      scf.condition(%more) %i, %carry : index, none
    } do {
    ^bb0(%i: index, %carry: none):
      %done = dataflow.graph.launch @work deps(%carry) values()
          stream_inputs() memories() stream_outputs() : (none) -> none
      %next = arith.addi %i, %one : index
      scf.yield %done, %next : none, index
    }
    dataflow.thread.yield %result#1 : none
  }

  dataflow.thread private @while_sync_forwarded(%limit: index)
      ctrl (%ctrl: none) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %result:2 = scf.while (%carry = %ctrl, %i = %zero)
        : (none, index) -> (index, none) {
      %forwarded = dataflow.sync %carry : (none) -> none
      %more = arith.cmpi ult, %i, %limit : index
      scf.condition(%more) %i, %forwarded : index, none
    } do {
    ^bb0(%i: index, %forwarded: none):
      %done = dataflow.graph.launch @work deps(%forwarded) values()
          stream_inputs() memories() stream_outputs() : (none) -> none
      %next = arith.addi %i, %one : index
      scf.yield %done, %next : none, index
    }
    dataflow.thread.yield %result#1 : none
  }

  dataflow.thread private @while_before_launch_forwarded(%limit: index)
      ctrl (%ctrl: none) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %result:2 = scf.while (%carry = %ctrl, %i = %zero)
        : (none, index) -> (index, none) {
      %done = dataflow.graph.launch @work deps(%carry) values()
          stream_inputs() memories() stream_outputs() : (none) -> none
      %more = arith.cmpi ult, %i, %limit : index
      scf.condition(%more) %i, %done : index, none
    } do {
    ^bb0(%i: index, %done: none):
      %next = arith.addi %i, %one : index
      scf.yield %done, %next : none, index
    }
    dataflow.thread.yield %result#1 : none
  }
}

//--- while-reordered-latest-only.mlir
module {
  dataflow.graph private @work(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @while_reordered_latest_only(%limit: index)
      ctrl (%ctrl: none) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %result:2 = scf.while (%i = %zero)
        : (index) -> (index, none) {
      %done = dataflow.graph.launch @work deps(%ctrl) values()
          stream_inputs() memories() stream_outputs() : (none) -> none
      %more = arith.cmpi ult, %i, %limit : index
      scf.condition(%more) %i, %done : index, none
    } do {
    ^bb0(%i: index, %latest: none):
      %next = arith.addi %i, %one : index
      scf.yield %next : index
    }
    dataflow.thread.yield %result#1 : none
  }
}

//--- for-valid.mlir
module {
  dataflow.graph private @work(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }

  dataflow.thread private @for_chained() ctrl (%ctrl: none) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %result = scf.for %i = %zero to %four step %one
        iter_args(%carry = %ctrl) -> (none) {
      %done = dataflow.graph.launch @work deps(%carry) values()
          stream_inputs() memories() stream_outputs() : (none) -> none
      scf.yield %done : none
    }
    dataflow.thread.yield %result : none
  }

  dataflow.thread private @for_aggregated() ctrl (%ctrl: none) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %result = scf.for %i = %zero to %four step %one
        iter_args(%aggregate = %ctrl) -> (none) {
      %done = dataflow.graph.launch @work deps(%ctrl) values()
          stream_inputs() memories() stream_outputs() : (none) -> none
      %joined:2 = dataflow.sync %aggregate, %done
          : (none, none) -> (none, none)
      scf.yield %joined#0 : none
    }
    dataflow.thread.yield %result : none
  }

  dataflow.thread private @for_zero_trip() ctrl (%ctrl: none) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %done = dataflow.graph.launch @work deps(%ctrl) values()
        stream_inputs() memories() stream_outputs() : (none) -> none
    %result = scf.for %i = %zero to %zero step %one
        iter_args(%carry = %done) -> (none) {
      scf.yield %carry : none
    }
    dataflow.thread.yield %result : none
  }
}

//--- for-latest-only.mlir
module {
  dataflow.graph private @work(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @for_latest_only() ctrl (%ctrl: none) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %result = scf.for %i = %zero to %four step %one
        iter_args(%latest = %ctrl) -> (none) {
      %done = dataflow.graph.launch @work deps(%ctrl) values()
          stream_inputs() memories() stream_outputs() : (none) -> none
      scf.yield %done : none
    }
    dataflow.thread.yield %result : none
  }
}

//--- repetitive-latest-only.mlir
module {
  dataflow.graph private @work(%start: none) -> ()
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @repetitive_latest_only(%limit: index)
      ctrl (%ctrl: none) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %result:2 = scf.while (%i = %zero, %latest = %ctrl)
        : (index, none) -> (index, none) {
      %more = arith.cmpi ult, %i, %limit : index
      scf.condition(%more) %i, %latest : index, none
    } do {
    ^bb0(%i: index, %latest: none):
      %done = dataflow.graph.launch @work deps(%ctrl) values()
          stream_inputs() memories() stream_outputs() : (none) -> none
      %next = arith.addi %i, %one : index
      scf.yield %next, %done : index, none
    }
    dataflow.thread.yield %result#1 : none
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

//--- nested-module.mlir
module {
  // Retirement is a property of the owning thread definition, not of the
  // module nesting depth at which that thread is defined.
  module {
    dataflow.graph private @work(%start: none) -> ()
        attributes {input_segments = array<i32: 0, 0, 0>,
                    result_segments = array<i32: 0, 0, 0>} {
      dataflow.graph.return %start : none
    }
    dataflow.thread private @nested_module() ctrl (%ctrl: none) {
      %done = dataflow.graph.launch @work deps(%ctrl) values()
          stream_inputs() memories() stream_outputs() : (none) -> none
      dataflow.thread.yield
    }
  }
}
