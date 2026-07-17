// RUN: loom-pnr-map --dfg-mlir %s --graph fresh_memory_export --hardware-mlir %S/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload fresh_memory_export --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON --implicit-check-not='"operation": "memref.alloc"' < %t.mapping.json

// CSV: fresh_memory_export,shared_reduction_adg,fresh_memory_export__fresh_memory_export__shared_reduction_adg,2,1,0,0,pass

// JSON-DAG: "status": "pass"
// JSON-DAG: "operation": "dataflow.store"
// JSON-NOT: "edge_ref": "memref.alloc
// JSON-NOT: "dataflow.store#0.operand0"

module {
  dataflow.graph.func private @fresh_memory_export(
      %start: none, %value: i32) -> (none, memref<1xi32>)
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 0, 1>} {
    %slot = memref.alloc() : memref<1xi32>
    %index = dataflow.constant %start {const_value = 0 : index} : index
    %done = dataflow.store %slot[%index] %value %start : memref<1xi32>
    dataflow.graph.return values() streams()
        memories(%slot : memref<1xi32>) complete(%done : none)
  }
}
