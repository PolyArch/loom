// RUN: loom %s -split-input-file -verify-diagnostics

fabric.module @alternating_routes(
    %a : !fabric.bits<32>, %b : !fabric.bits<32>,
    %spare : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                      %pb = %b : !fabric.bits<32>,
                      %ps = %spare : !fabric.bits<32>) -> !fabric.bits<32> {
    %r = fabric.fu(%x = %pa : !fabric.bits<32>,
                   %y = %pb : !fabric.bits<32>,
                   %unused = %ps : !fabric.bits<32>) -> !fabric.bits<32>
        attributes {valid_encodings = [{outputs = [0 : i32], resources = [
          {resource = 0 : i32, mode = 0 : i32},
          {resource = 1 : i32, select = 0 : i32},
          {resource = 2 : i32, select = 0 : i32},
          {resource = 3 : i32, select = 1 : i32},
          {resource = 4 : i32, select = 0 : i32}
        ]}]} {
      %sum = fabric.op [@arith.addi] (%x, %y)
             {hw_params = [{op = @arith.addi,
               function_type = (i32, i32) -> i32,
               input_ports = [0 : i32, 1 : i32],
               output_ports = [0 : i32], attributes = {}}]}
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      %d0, %d1 = fabric.demux %sum : !fabric.bits<32> -> 2
      %m0 = fabric.mux %d0, %unused : !fabric.bits<32>
      %e0, %e1 = fabric.demux %m0 : !fabric.bits<32> -> 2
      %m1 = fabric.mux %e1, %unused : !fabric.bits<32>
      fabric.yield %m1 : !fabric.bits<32>
    }
  }
  fabric.yield
}

// -----

fabric.module @route_only_cycle(
    %a : !fabric.bits<32>, %b : !fabric.bits<32>,
    %spare : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                      %pb = %b : !fabric.bits<32>,
                      %ps = %spare : !fabric.bits<32>)
                     -> (!fabric.bits<32>, !fabric.bits<32>) {
    // expected-error @+1 {{valid semantic encoding #0: selected routing topology contains a cycle}}
    %r:2 = fabric.fu(%x = %pa : !fabric.bits<32>,
                     %y = %pb : !fabric.bits<32>,
                     %unused = %ps : !fabric.bits<32>)
                    -> (!fabric.bits<32>, !fabric.bits<32>)
        attributes {valid_encodings = [{outputs = [0 : i32, 1 : i32],
          resources = [
            {resource = 0 : i32, select = 0 : i32},
            {resource = 1 : i32, select = 0 : i32},
            {resource = 2 : i32, mode = 0 : i32}
          ]}]} {
      %m = fabric.mux %d0, %unused : !fabric.bits<32>
      %d0, %d1 = fabric.demux %m : !fabric.bits<32> -> 2
      %sum = fabric.op [@arith.addi] (%x, %y)
             {hw_params = [{op = @arith.addi,
               function_type = (i32, i32) -> i32,
               input_ports = [0 : i32, 1 : i32],
               output_ports = [0 : i32], attributes = {}}]}
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %m, %sum : !fabric.bits<32>, !fabric.bits<32>
    }
  }
  fabric.yield
}
