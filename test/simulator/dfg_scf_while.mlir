// RUN: loom-dfg-sim %s --graph structured_while_pointer_min --arg 0=none --arg 1=0 --arg 2=3 --arg 3=1 --memref 4=3,-4,7 --memref 5=2,-5,9 --memref 6=0,0,0 --output %t.min.json
// RUN: FileCheck %s --check-prefix=MIN < %t.min.json

// MIN-DAG: "workload": "structured_while_pointer_min"
// MIN-DAG: "graph": "structured_while_pointer_min"
// MIN-DAG: "status": "pass"
// MIN-DAG: "dynamic_work_items": 3
// MIN-DAG: "dataflow.load": 6
// MIN-DAG: "llvm.intr.smin": 3
// MIN-DAG: "dataflow.store": 3
// MIN-DAG: "final_outputs": [
// MIN-DAG: "none"
// MIN-DAG: "arg6": [
// MIN-DAG: "i8:2"
// MIN-DAG: "i8:-5"
// MIN-DAG: "i8:7"

module {
  dataflow.graph.func private @structured_while_pointer_min(
      %ctrl: none, %iv0: i32, %ub: i32, %step: i32,
      %lhs: memref<?xi8>, %rhs: memref<?xi8>, %out: memref<?xi8>) -> none {
    %done = scf.while (%iv = %iv0) : (i32) -> i32 {
      %idx = arith.index_cast %iv : i32 to index
      %ldata, %ldone = dataflow.load %lhs[%idx] %ctrl : memref<?xi8>
      %rdata, %rdone = dataflow.load %rhs[%idx] %ctrl : memref<?xi8>
      %selected = llvm.intr.smin(%ldata, %rdata) : (i8, i8) -> i8
      %sdone = dataflow.store %out[%idx] %selected %ctrl : memref<?xi8>
      %next = arith.addi %iv, %step : i32
      %cont = arith.cmpi slt, %next, %ub : i32
      scf.condition(%cont) %next : i32
    } do {
    ^bb0(%next: i32):
      scf.yield %next : i32
    }
    dataflow.graph.return %ctrl : none
  }
}
