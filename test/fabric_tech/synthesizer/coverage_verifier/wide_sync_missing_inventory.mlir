// RUN: loom-coverage-test %s | FileCheck %s

fabric.module @exact_sync(
    %a: !fabric.bits<32>, %b: !fabric.bits<32>,
    %c: !fabric.bits<32>, %d: !fabric.bits<32>) {
  fabric.pe [spatial] (
      %pa = %a : !fabric.bits<32>, %pb = %b : !fabric.bits<32>,
      %pc = %c : !fabric.bits<32>, %pd = %d : !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>) {
    %r:4 = fabric.fu(
        %x0 = %pa : !fabric.bits<32>, %x1 = %pb : !fabric.bits<32>,
        %x2 = %pc : !fabric.bits<32>, %x3 = %pd : !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>,
            !fabric.bits<32>, !fabric.bits<32>)
        attributes {valid_encodings = [{
          outputs = [0 : i32, 1 : i32, 2 : i32, 3 : i32],
          resources = [{resource = 0 : i32, mode = 0 : i32}]
        }]} {
      %y0, %y1, %y2, %y3 = fabric.op [@dataflow.sync]
          (%x0, %x1, %x2, %x3) {hw_params = [{
            op = @dataflow.sync,
            function_type = (i32, i32, i32, i32) ->
                            (i32, i32, i32, i32),
            input_ports = [0 : i32, 1 : i32, 2 : i32, 3 : i32],
            output_ports = [0 : i32, 1 : i32, 2 : i32, 3 : i32],
            attributes = {}
          }]} : (!fabric.bits<32>, !fabric.bits<32>,
                 !fabric.bits<32>, !fabric.bits<32>)
                -> (!fabric.bits<32>, !fabric.bits<32>,
                    !fabric.bits<32>, !fabric.bits<32>)
      fabric.yield %y0, %y1, %y2, %y3 :
          !fabric.bits<32>, !fabric.bits<32>,
          !fabric.bits<32>, !fabric.bits<32>
    }
  }
  fabric.yield
}

func.func @subset_without_inventory(%a: i32, %b: i32) -> i32
    attributes {loom.coverage_input = true} {
  %x, %y = dataflow.sync %a, %b : (i32, i32) -> (i32, i32)
  return %x : i32
}

// CHECK: coverage[0] funcname=subset_without_inventory matched=false index=none
// CHECK-NEXT: all_covered=false
