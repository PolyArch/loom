// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: not loom-dfg-sim %t.dir/memref.mlir --graph memref_rejects_offset --memref 0:4=1.000000e+00 --output %t.bad.json 2>&1 | FileCheck %s --check-prefix=MEMREF-ERR

// MEMREF-ERR: memref argument 0 cannot use a nonzero memory fixture byte offset

//--- memref.mlir
module {
  dataflow.graph private @memref_rejects_offset(
      %ctrl: none, %mem: memref<?xf32>) -> ()
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %ctrl : none
  }
}
