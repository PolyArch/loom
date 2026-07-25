// RUN: loom-dfg-sim %s --graph duplicate_scatter --arg 0=8589934593 \
// RUN:   --arg 1=0x0000000200000002 --memref 2=10,11,12,13 \
// RUN:   --output %t.duplicate.json
// RUN: FileCheck %s --check-prefix=DUPLICATE < %t.duplicate.json

// docs/spec-dataflow-vectorization.md gives a plain scatter no lane order for
// duplicate active addresses: the finalized program must already have proved
// them distinct or lowered the access to an explicit program order. A finalized
// actor that still resolves duplicates therefore breaks an invariant its own
// provider guarantees, which docs/spec-simulation-artifacts.md classifies as
// ExecutionFailed rather than as invalid IR or a missing capability. The
// refused firing publishes nothing, so the run exports no outputs, no memory
// state, and no memory-root result.
// DUPLICATE-DAG: "final_memory_roots": {}
// DUPLICATE-DAG: "final_memory_state": {}
// DUPLICATE-DAG: "final_outputs": []
// DUPLICATE-NOT: "dataflow.store":
// DUPLICATE: "status": "execution_failed"

module {
  module attributes {
    dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
  } {
    dataflow.graph private @duplicate_scatter(
        %start: none, %packed: i64, %addresses: vector<2xindex>,
        %mem: memref<4xi32>) -> ()
        attributes {input_segments = array<i32: 2, 0, 1>,
                    result_segments = array<i32: 0, 0, 0>} {
      %data = dataflow.unpack %packed : i64 -> vector<2xi32>
      %store_done = dataflow.store %mem[%addresses] %data %start
          : memref<4xi32>, vector<2xindex>, vector<2xi32>
      dataflow.graph.return values() streams() memories()
          complete(%store_done : none)
    }
  }
}
