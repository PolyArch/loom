// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-for-to-graph %t.dir/supported.mlir | FileCheck %s --check-prefix=SUPPORTED
// RUN: not loom-raise-opt --loom-lower-for-to-graph --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/atomic-invalid.mlir 2>&1 | FileCheck %s --check-prefix=ATOMIC --implicit-check-not="dataflow.graph private" --implicit-check-not=dataflow.graph.launch

// SUPPORTED-LABEL: dataflow.thread private @for_completion
// SUPPORTED-SAME: ctrl (%[[FOR_START:.*]]: none)
// SUPPORTED: %[[FOR_DONE:.*]] = scf.for
// SUPPORTED-SAME: iter_args(%[[FOR_CARRY:.*]] = %[[FOR_START]]) -> (none)
// SUPPORTED: %[[FOR_SELECTED:.*]] = scf.if %{{.*}} -> (none)
// SUPPORTED: %[[FOR_LAUNCH:.*]] = dataflow.graph.launch @for_graph deps(%[[FOR_CARRY]])
// SUPPORTED: scf.yield %[[FOR_LAUNCH]] : none
// SUPPORTED: scf.yield %[[FOR_CARRY]] : none
// SUPPORTED: scf.yield %[[FOR_SELECTED]] : none
// SUPPORTED: dataflow.thread.yield %[[FOR_DONE]] : none

// SUPPORTED-LABEL: dataflow.thread private @while_completion
// SUPPORTED-SAME: ctrl (%[[WHILE_START:.*]]: none)
// SUPPORTED: %[[WHILE_DONE:.*]]:2 = scf.while
// SUPPORTED-SAME: = %[[WHILE_START]]
// SUPPORTED-SAME: = %[[WHILE_START]]
// SUPPORTED: %[[BEFORE_LAUNCH:.*]] = dataflow.graph.launch @while_before_graph deps(%[[BEFORE_CARRY:.*]])
// SUPPORTED: scf.condition(%{{.*}}) %[[BEFORE_LAUNCH]], %[[AFTER_CARRY:.*]] : none, none
// SUPPORTED: ^bb0(%[[BEFORE_FEEDBACK:.*]]: none, %[[AFTER_INPUT:.*]]: none):
// SUPPORTED: %[[AFTER_LAUNCH:.*]] = dataflow.graph.launch @while_after_graph deps(%[[AFTER_INPUT]])
// SUPPORTED: scf.yield %[[BEFORE_FEEDBACK]], %[[AFTER_LAUNCH]] : none, none
// SUPPORTED: dataflow.thread.yield %[[WHILE_DONE]]#0, %[[WHILE_DONE]]#1 : none, none

// SUPPORTED-LABEL: dataflow.thread private @switch_completion
// SUPPORTED-SAME: ctrl (%[[SWITCH_START:.*]]: none)
// SUPPORTED: %[[SWITCH_DONE:.*]] = scf.index_switch
// SUPPORTED: case 7 {
// SUPPORTED: %[[SWITCH_LAUNCH:.*]] = dataflow.graph.launch @switch_graph deps(%[[SWITCH_START]])
// SUPPORTED: scf.yield %[[SWITCH_LAUNCH]] : none
// SUPPORTED: default {
// SUPPORTED: scf.yield %[[SWITCH_START]] : none
// SUPPORTED: dataflow.thread.yield %[[SWITCH_DONE]] : none

// SUPPORTED-LABEL: dataflow.thread private @parallel_completion
// SUPPORTED-SAME: ctrl (%[[PAR_START:.*]]: none)
// SUPPORTED-DAG: %[[PAR_LAUNCH0:.*]] = dataflow.graph.launch @parallel_graph deps(%[[PAR_START]])
// SUPPORTED-DAG: %[[PAR_LAUNCH1:.*]] = dataflow.graph.launch @parallel_graph deps(%[[PAR_START]])
// SUPPORTED: %[[PAR_ALL:.*]]:2 = dataflow.sync %[[PAR_LAUNCH0]], %[[PAR_LAUNCH1]]
// SUPPORTED: dataflow.thread.yield %[[PAR_ALL]]#0 : none

// SUPPORTED-LABEL: dataflow.thread private @forall_completion
// SUPPORTED-SAME: ctrl (%[[FORALL_START:.*]]: none)
// SUPPORTED-DAG: %[[FORALL_LAUNCH0:.*]] = dataflow.graph.launch @forall_graph deps(%[[FORALL_START]])
// SUPPORTED-DAG: %[[FORALL_LAUNCH1:.*]] = dataflow.graph.launch @forall_graph deps(%[[FORALL_START]])
// SUPPORTED: %[[FORALL_ALL:.*]]:2 = dataflow.sync %[[FORALL_LAUNCH0]], %[[FORALL_LAUNCH1]]
// SUPPORTED: dataflow.thread.yield %[[FORALL_ALL]]#0 : none

// ATOMIC: error: {{.*}}completion propagation through enclosing 'scf.execute_region'
// ATOMIC-LABEL: dataflow.thread private @publishable
// ATOMIC: loom.spatial_region
// ATOMIC-LABEL: dataflow.thread private @unsupported
// ATOMIC: scf.execute_region
// ATOMIC: loom.spatial_region

//--- supported.mlir
module {
  dataflow.thread private @for_completion(%limit: index, %enabled: i1)
      ctrl (%start: none) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %limit step %c1 {
      scf.if %enabled {
        "loom.spatial_region"()
            <{operandSegmentSizes = array<i32: 0, 0, 0, 0>,
              resultSegmentSizes = array<i32: 0, 0>}> ({
          ^bb0:
            "loom.spatial_yield"()
                <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
        }) {graph_name = "for_graph", source_maps = []} : () -> ()
      }
    }
    dataflow.thread.yield
  }

  dataflow.thread private @while_completion(%continue: i1)
      ctrl (%start: none) {
    scf.while : () -> () {
      "loom.spatial_region"()
          <{operandSegmentSizes = array<i32: 0, 0, 0, 0>,
            resultSegmentSizes = array<i32: 0, 0>}> ({
        ^bb0:
          "loom.spatial_yield"()
              <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
      }) {graph_name = "while_before_graph", source_maps = []} : () -> ()
      scf.condition(%continue)
    } do {
      "loom.spatial_region"()
          <{operandSegmentSizes = array<i32: 0, 0, 0, 0>,
            resultSegmentSizes = array<i32: 0, 0>}> ({
        ^bb0:
          "loom.spatial_yield"()
              <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
      }) {graph_name = "while_after_graph", source_maps = []} : () -> ()
      scf.yield
    }
    dataflow.thread.yield
  }

  dataflow.thread private @switch_completion(%selector: index)
      ctrl (%start: none) {
    scf.index_switch %selector
    case 7 {
      "loom.spatial_region"()
          <{operandSegmentSizes = array<i32: 0, 0, 0, 0>,
            resultSegmentSizes = array<i32: 0, 0>}> ({
        ^bb0:
          "loom.spatial_yield"()
              <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
      }) {graph_name = "switch_graph", source_maps = []} : () -> ()
      scf.yield
    }
    default {
      scf.yield
    }
    dataflow.thread.yield
  }

  dataflow.thread private @parallel_completion() ctrl (%start: none) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
      "loom.spatial_region"()
          <{operandSegmentSizes = array<i32: 0, 0, 0, 0>,
            resultSegmentSizes = array<i32: 0, 0>}> ({
        ^bb0:
          "loom.spatial_yield"()
              <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
      }) {graph_name = "parallel_graph", source_maps = []} : () -> ()
      scf.reduce
    } {loom.parallel_group = 0 : i64}
    dataflow.thread.yield
  }

  dataflow.thread private @forall_completion() ctrl (%start: none) {
    scf.forall (%i) in (2) {
      "loom.spatial_region"()
          <{operandSegmentSizes = array<i32: 0, 0, 0, 0>,
            resultSegmentSizes = array<i32: 0, 0>}> ({
        ^bb0:
          "loom.spatial_yield"()
              <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
      }) {graph_name = "forall_graph", source_maps = []} : () -> ()
    } {loom.parallel_group = 1 : i64}
    dataflow.thread.yield
  }
}

//--- atomic-invalid.mlir
module {
  dataflow.thread private @publishable() ctrl (%start: none) {
    "loom.spatial_region"()
        <{operandSegmentSizes = array<i32: 0, 0, 0, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0:
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "publishable_graph", source_maps = []} : () -> ()
    dataflow.thread.yield
  }

  dataflow.thread private @unsupported() ctrl (%start: none) {
    scf.execute_region {
      "loom.spatial_region"()
          <{operandSegmentSizes = array<i32: 0, 0, 0, 0>,
            resultSegmentSizes = array<i32: 0, 0>}> ({
        ^bb0:
          "loom.spatial_yield"()
              <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
      }) {graph_name = "unsupported_graph", source_maps = []} : () -> ()
      scf.yield
    }
    dataflow.thread.yield
  }
}
