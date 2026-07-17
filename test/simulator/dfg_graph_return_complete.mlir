// RUN: loom-dfg-sim %s --graph multi_complete --arg 0=none --arg 1=none --arg 2=7 --output %t.json
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
      %start: none, %other: none, %value: i32) -> (none, i32) {
    dataflow.graph.return values(%value : i32) streams() memories()
        complete(%start, %other : none, none)
  }
}
