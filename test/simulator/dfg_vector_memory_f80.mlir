// RUN: loom-dfg-sim %s --graph f80_memory_roundtrip \
// RUN:   --arg 0=0xFEDCBA9876543210ABCD \
// RUN:   --arg 1=0x89ABCDEF0123456789ABFEDCBA9876543210ABCD \
// RUN:   --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK: "event_count": 5
// CHECK: "final_outputs": [
// CHECK-NEXT: "none",
// CHECK-NEXT: "vector<1xf80>:0xFEDCBA9876543210ABCD",
// CHECK-NEXT: "vector<2xf80>:0x89ABCDEF0123456789ABFEDCBA9876543210ABCD"
// CHECK: "dataflow.constant": 1
// CHECK-NEXT: "dataflow.load": 2
// CHECK-NEXT: "dataflow.store": 2
// CHECK: "status": "pass"

module {
  dataflow.graph private @f80_memory_roundtrip(
      %start: none, %scalar_value: vector<1xf80>,
      %contiguous_value: vector<2xf80>) -> (vector<1xf80>, vector<2xf80>)
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 2, 0, 0>} {
    %idx = dataflow.constant %start {const_value = 0 : index} : index
    %scalar_mem = memref.alloc() : memref<1xvector<1xf80>>
    %contiguous_mem = memref.alloc() : memref<2xf80>

    %scalar_stored =
        dataflow.store %scalar_mem[%idx] %scalar_value %start
            : memref<1xvector<1xf80>>
    %scalar_loaded, %scalar_done =
        dataflow.load %scalar_mem[%idx] %scalar_stored
            : memref<1xvector<1xf80>>

    %contiguous_stored =
        dataflow.store %contiguous_mem[%idx] %contiguous_value %scalar_done
            : memref<2xf80>, vector<2xf80>
    %contiguous_loaded, %done =
        dataflow.load %contiguous_mem[%idx] %contiguous_stored
            : memref<2xf80>, vector<2xf80>

    dataflow.graph.return %done, %scalar_loaded, %contiguous_loaded
        : none, vector<1xf80>, vector<2xf80>
  }
}
