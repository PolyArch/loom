// RUN: not loom-dfg-sim %s --graph duplicate_scatter --arg 0=8589934593 \
// RUN:   --memref 1=10,11,12,13 --output %t.duplicate.json 2>&1 \
// RUN:   | FileCheck %s

// docs/spec-dataflow-vectorization.md gives a plain scatter no lane order for
// duplicate active addresses: the finalized program must already have proved
// them distinct or lowered the access to an explicit program order. A finalized
// actor with statically duplicated addresses is rejected before canonical
// Artifact publication. Runtime ProviderInvariant remains reserved for a
// provider whose resolved behavior contradicts this finalizer guarantee.
// CHECK: plain scatter active-address distinctness proof not established

module {
  module attributes {
    dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
  } {
    dataflow.graph private @duplicate_scatter(
        %start: none, %packed: i64, %mem: memref<4xi32>) -> ()
        attributes {input_segments = array<i32: 1, 0, 1>,
                    result_segments = array<i32: 0, 0, 0>} {
      %addresses = arith.constant dense<[2, 2]> : vector<2xindex>
      %data = dataflow.unpack %packed : i64 -> vector<2xi32>
      %store_done = dataflow.store %mem[%addresses] %data %start
          : memref<4xi32>, vector<2xindex>, vector<2xi32>
      dataflow.graph.return values() streams() memories()
          complete(%store_done : none)
    }
  }
}
