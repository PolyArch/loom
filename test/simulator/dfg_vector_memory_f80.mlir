// RUN: loom-dfg-sim %s --graph f80_memory_roundtrip \
// RUN:   --arg 0=0xFEDCBA9876543210ABCD \
// RUN:   --arg 1=0x89ABCDEF0123456789ABFEDCBA9876543210ABCD \
// RUN:   --output %t.json
// RUN: FileCheck %s < %t.json
// RUN: loom-dfg-sim %s --graph scalar_f80_memory_reentry --invocations 2 \
// RUN:   --memref 0=0xFEDCBA9876543210ABCD --output %t.scalar.json
// RUN: FileCheck %s --check-prefix=SCALAR < %t.scalar.json
// RUN: not loom-dfg-sim %s --graph scalar_f80_memory_reentry \
// RUN:   --memref 0=0xEDCBA9876543210ABCD --output %t.short.json 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SHORT
// RUN: not loom-dfg-sim %s --graph scalar_f80_memory_reentry \
// RUN:   --memref 0=0xFEDCBA9876543210ABCZ --output %t.malformed.json 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MALFORMED

// CHECK: "event_count": 6
// CHECK: "final_outputs": [
// CHECK-NEXT: "none",
// CHECK-NEXT: "vector<1xf80>:0xFEDCBA9876543210ABCD",
// CHECK-NEXT: "vector<2xf80>:0x89ABCDEF0123456789ABFEDCBA9876543210ABCD",
// CHECK-NEXT: "f80:0xFEDCBA9876543210ABCD"
// CHECK: "dataflow.constant": 1
// CHECK-NEXT: "dataflow.load": 3
// CHECK-NEXT: "dataflow.store": 2
// CHECK: "status": "pass"

// SCALAR: "event_count": 6
// SCALAR: "arg0": [
// SCALAR-NEXT: "f80:0xFEDCBA9876543210ABCD"
// SCALAR: "final_outputs": [
// SCALAR-NEXT: "none"
// SCALAR: "dataflow.constant": 2
// SCALAR-NEXT: "dataflow.load": 2
// SCALAR-NEXT: "dataflow.store": 2
// SCALAR: "status": "pass"

// SHORT: exact f80 argument requires 20 hexadecimal digits
// MALFORMED: exact f80 argument is not canonical hexadecimal

module {
  dataflow.graph private @f80_memory_roundtrip(
      %start: none, %scalar_value: vector<1xf80>,
      %contiguous_value: vector<2xf80>)
      -> (vector<1xf80>, vector<2xf80>, f80)
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 3, 0, 0>} {
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
    %contiguous_loaded, %contiguous_done =
        dataflow.load %contiguous_mem[%idx] %contiguous_stored
            : memref<2xf80>, vector<2xf80>
    %lane_zero, %done =
        dataflow.load %contiguous_mem[%idx] %contiguous_done
            : memref<2xf80>

    dataflow.graph.return
        %done, %scalar_loaded, %contiguous_loaded, %lane_zero
        : none, vector<1xf80>, vector<2xf80>, f80
  }

  dataflow.graph private @scalar_f80_memory_reentry(
      %start: none, %memory: memref<1xf80>) -> (memref<1xf80>)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 1>} {
    %idx = dataflow.constant %start {const_value = 0 : index} : index
    %loaded, %load_done =
        dataflow.load %memory[%idx] %start : memref<1xf80>
    %done =
        dataflow.store %memory[%idx] %loaded %load_done : memref<1xf80>
    dataflow.graph.return values() streams()
        memories(%memory : memref<1xf80>) complete(%done : none)
  }
}
