// RUN: loom-dfg-sim %s --graph allocated_vector_element \
// RUN:   --arg 0=1144201745 --output %t.alloc.json
// RUN: FileCheck %s --check-prefix=ALLOC < %t.alloc.json
// RUN: loom-dfg-sim %s --graph offset_vector_element \
// RUN:   --arg 0=0x44332211 \
// RUN:   --memref 1:4=0x04030201,0x08070605 --output %t.offset.json
// RUN: FileCheck %s --check-prefix=OFFSET < %t.offset.json

// ALLOC-DAG: "status": "pass"
// ALLOC-DAG: "dataflow.load": 1
// ALLOC-DAG: "dataflow.store": 1
// ALLOC-DAG: "vector<4xi8>:0x44332211"

// OFFSET: "event_count": 3
// OFFSET: "arg1": [
// OFFSET-NEXT: "vector<4xi8>:0x4030201",
// OFFSET-NEXT: "vector<4xi8>:0x44332211"
// OFFSET: "final_outputs": [
// OFFSET-NEXT: "none",
// OFFSET-NEXT: "vector<4xi8>:0x8070605"
// OFFSET: "dataflow.load": 1
// OFFSET-NEXT: "dataflow.store": 1
// OFFSET: "status": "pass"

module {
  dataflow.graph private @allocated_vector_element(
      %start: none, %packed: i32) -> vector<4xi8>
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value = dataflow.unpack %packed : i32 -> vector<4xi8>
    %idx = dataflow.constant %start {const_value = 0 : index} : index
    %slot = memref.alloc() : memref<1xvector<4xi8>>
    %stored = dataflow.store %slot[%idx] %value %start
        : memref<1xvector<4xi8>>
    %loaded, %done = dataflow.load %slot[%idx] %stored
        : memref<1xvector<4xi8>>
    dataflow.graph.return %done, %loaded : none, vector<4xi8>
  }

  dataflow.graph private @offset_vector_element(
      %start: none, %value: vector<4xi8>, %ptr: !llvm.ptr)
      -> vector<4xi8>
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %idx = dataflow.constant %start {const_value = 0 : index} : index
    %mem = builtin.unrealized_conversion_cast %ptr
        : !llvm.ptr to memref<?xvector<4xi8>>
    %loaded, %read = dataflow.load %mem[%idx] %start
        : memref<?xvector<4xi8>>
    %done = dataflow.store %mem[%idx] %value %read
        : memref<?xvector<4xi8>>
    dataflow.graph.return %done, %loaded : none, vector<4xi8>
  }
}
