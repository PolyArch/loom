// RUN: loom-dfg-sim %s --graph multi_complete --arg 0=7 --output %t.json
// RUN: FileCheck %s < %t.json

// A variadic complete segment is one unordered all-of done result at the graph
// ABI. The simulator must require every witness without reporting each witness
// as a separate output.

// CHECK: "final_outputs": [
// CHECK-NEXT: "none",
// CHECK-NEXT: "i32:7"
// CHECK-NEXT: ]
// CHECK: "status": "pass"
module {
  dataflow.graph.func private @multi_complete(
      %start: none, %value: i32) -> (none, i32)
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %done:2 = dataflow.sync %start, %start
        : (none, none) -> (none, none)
    %published:2 = dataflow.sync %done#0, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%published#1 : i32) streams() memories()
        complete(%published#0, %done#1 : none, none)
  }
}
