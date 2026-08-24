// RUN: not loom-dfg-sim %s --graph unknown_scatter \
// RUN:   --arg 0=0x0000000100000000 --arg 1=0x0000000200000001 \
// RUN:   --memref 2=0,0,0,0 --output %t.json 2>&1 | FileCheck %s

// A graph input has no canonical guarantee that its active address lanes are
// distinct. Supplying distinct values for one invocation cannot establish the
// program-wide plain-scatter contract.
// CHECK: plain scatter active-address distinctness proof not established

module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  dataflow.graph private @unknown_scatter(
      %start: none, %addresses: vector<2xindex>, %data: vector<2xi32>,
      %memory: memref<4xi32>)
      attributes {input_segments = array<i32: 2, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %done = dataflow.store %memory[%addresses] %data %start
        : memref<4xi32>, vector<2xindex>, vector<2xi32>
    dataflow.graph.return values() streams() memories()
        complete(%done : none)
  }
}
