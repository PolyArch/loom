// RUN: loom %s | FileCheck %s
// RUN: loom %s | loom | FileCheck %s
// RUN: loom %s --emit-bytecode | loom | FileCheck %s
// RUN: loom %s --mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC

// The protocol start argument is not an application ABI slot. Application
// argument and result dictionaries stay attached to their function-interface
// indices through textual and bytecode serialization.
// CHECK-LABEL: dataflow.graph private @interface_metadata(
// CHECK-SAME: %{{.*}}: none,
// CHECK-SAME: %{{.*}}: i32 {test.role = "value"},
// CHECK-SAME: %{{.*}}: memref<?xi32> {llvm.noalias, test.argument = 7 : i32})
// CHECK-SAME: -> (i32 {test.result = "preserved"})
// GENERIC: "dataflow.graph"()
// GENERIC-SAME: arg_attrs = [{test.role = "value"}, {llvm.noalias, test.argument = 7 : i32}]
// GENERIC-SAME: function_type = (i32, memref<?xi32>) -> i32
// GENERIC-SAME: res_attrs = [{test.result = "preserved"}]
dataflow.graph private @interface_metadata(
    %start: none,
    %value: i32 {test.role = "value"},
    %memory: memref<?xi32> {llvm.noalias, test.argument = 7 : i32})
    -> (i32 {test.result = "preserved"})
    attributes {input_segments = array<i32: 1, 0, 1>,
                result_segments = array<i32: 1, 0, 0>} {
  dataflow.graph.return %start, %value : none, i32
}
