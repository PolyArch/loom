// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: not loom-pnr-map --dfg-mlir %t.lowered.mlir --graph normalized_add \
// RUN:   --hardware-mlir %s --hardware normalized_add_adg \
// RUN:   --workload normalized_add --output %t.csv --artifact %t.json 2>&1 \
// RUN:   | FileCheck %s

// CHECK: legacy PnR cannot consume normalized fabric.op hw_params
// CHECK-SAME: selected fabric.fu semantic encoding is required

module {
  dataflow.graph.func private @normalized_add(
      %ctrl: none, %lhs: i32, %rhs: i32) -> (none, i32) {
    %sum = arith.addi %lhs, %rhs : i32
    dataflow.graph.return %ctrl, %sum : none, i32
  }

  fabric.module @normalized_add_adg(%lhs : !fabric.bits<32>,
                                    %rhs : !fabric.bits<32>) {
    %sum = fabric.pe [spatial] (%pa = %lhs : !fabric.bits<32>,
                                %pb = %rhs : !fabric.bits<32>)
        -> !fabric.bits<32> {
      %result = fabric.fu(%fa = %pa : !fabric.bits<32>,
                          %fb = %pb : !fabric.bits<32>)
          -> !fabric.bits<32>
          attributes {valid_encodings = [{
            outputs = [0 : i32],
            resources = [{resource = 0 : i32, mode = 0 : i32}]
          }]} {
        %value = fabric.op [@arith.addi] (%fa, %fb)
            {hw_params = [{
              op = @arith.addi,
              function_type = (i32, i32) -> i32,
              input_ports = [0 : i32, 1 : i32],
              output_ports = [0 : i32],
              attributes = {}
            }]}
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %value : !fabric.bits<32>
      }
    }
    fabric.yield
  }
}
