// RUN: loom %s -verify-diagnostics

fabric.module @width_too_narrow(%a : !fabric.bits<8>,
                                %b : !fabric.bits<8>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<8>,
                      %pb = %b : !fabric.bits<8>) -> !fabric.bits<8> {
    %r = fabric.fu(%x = %pa : !fabric.bits<8>,
                   %y = %pb : !fabric.bits<8>) -> !fabric.bits<8>
        attributes {valid_encodings = [{outputs = [0 : i32], resources = [
          {resource = 0 : i32, mode = 0 : i32}
        ]}]} {
      // expected-error @+1 {{software input type width exceeds the physical payload width}}
      %v = fabric.op [@arith.addi] (%x, %y)
           {hw_params = [{op = @arith.addi,
             function_type = (i16, i16) -> i16,
             input_ports = [0 : i32, 1 : i32],
             output_ports = [0 : i32], attributes = {}}]}
           : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %v : !fabric.bits<8>
    }
  }
  fabric.yield
}

// -----

fabric.module @vector_width_overflow(%a : !fabric.bits<0>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<0>) -> !fabric.bits<0> {
    %r = fabric.fu(%x = %pa : !fabric.bits<0>) -> !fabric.bits<0>
        attributes {valid_encodings = [{outputs = [0 : i32], resources = [
          {resource = 0 : i32, mode = 0 : i32}
        ]}]} {
      // expected-error @+1 {{vector bit width exceeds unsigned range}}
      %v = fabric.op [@arith.addi] (%x, %x)
           {hw_params = [{op = @arith.addi,
             function_type = (vector<65536x65536xi8>,
                              vector<65536x65536xi8>) -> vector<65536x65536xi8>,
             input_ports = [0 : i32, 1 : i32],
             output_ports = [0 : i32], attributes = {}}]}
           : (!fabric.bits<0>, !fabric.bits<0>) -> !fabric.bits<0>
      fabric.yield %v : !fabric.bits<0>
    }
  }
  fabric.yield
}

// -----

fabric.module @vector_rank_zero(%a : !fabric.bits<8>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<8>) -> !fabric.bits<8> {
    %r = fabric.fu(%x = %pa : !fabric.bits<8>) -> !fabric.bits<8>
        attributes {valid_encodings = [{outputs = [0 : i32], resources = [
          {resource = 0 : i32, mode = 0 : i32}
        ]}]} {
      // expected-error @+1 {{rank-zero vector has no fixed bit width}}
      %v = fabric.op [@arith.addi] (%x, %x)
           {hw_params = [{op = @arith.addi,
             function_type = (vector<i8>, vector<i8>) -> vector<i8>,
             input_ports = [0 : i32, 1 : i32],
             output_ports = [0 : i32], attributes = {}}]}
           : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %v : !fabric.bits<8>
    }
  }
  fabric.yield
}
