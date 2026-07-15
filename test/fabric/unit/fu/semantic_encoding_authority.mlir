// RUN: loom %s -verify-diagnostics

fabric.module @op_list_must_match_modes(%a : !fabric.bits<32>,
                                        %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                      %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    %r = fabric.fu(%x = %pa : !fabric.bits<32>,
                   %y = %pb : !fabric.bits<32>) -> !fabric.bits<32>
        attributes {valid_encodings = [{outputs = [0 : i32], resources = [
          {resource = 0 : i32, mode = 0 : i32}
        ]}]} {
      // expected-error @+1 {{op_list operation @arith.subi has no hw_params mode}}
      %v = fabric.op [@arith.addi, @arith.subi] (%x, %y)
           {hw_params = [{op = @arith.addi,
             function_type = (i32, i32) -> i32,
             input_ports = [0 : i32, 1 : i32],
             output_ports = [0 : i32], attributes = {}}]}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

fabric.module @canonical_rejects_legacy_hw_params(%a : !fabric.bits<32>,
                                                  %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                      %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    %r = fabric.fu(%x = %pa : !fabric.bits<32>,
                   %y = %pb : !fabric.bits<32>) -> !fabric.bits<32>
        attributes {valid_encodings = [{outputs = [0 : i32], resources = [
          {resource = 0 : i32, mode = 0 : i32}
        ]}]} {
      // expected-error @+1 {{hw_params mode #0: hw_params mode requires op, function_type, and attributes}}
      %v = fabric.op [@arith.addi] (%x, %y)
           {hw_params = [{}]}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

fabric.module @mode_must_be_valid_software_op(%a : !fabric.bits<32>,
                                              %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                      %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    %r = fabric.fu(%x = %pa : !fabric.bits<32>,
                   %y = %pb : !fabric.bits<32>) -> !fabric.bits<32>
        attributes {valid_encodings = [{outputs = [0 : i32], resources = [
          {resource = 0 : i32, mode = 0 : i32}
        ]}]} {
      // expected-error @+1 {{hw_params mode #0 does not form a valid @arith.addi operation}}
      %v = fabric.op [@arith.addi] (%x, %y)
           {hw_params = [{op = @arith.addi,
             function_type = (f32, f32) -> f32,
             input_ports = [0 : i32, 1 : i32],
             output_ports = [0 : i32], attributes = {}}]}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}

fabric.module @pseudo_intrinsic_is_not_a_materializable_mode(
    %a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                      %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    %r = fabric.fu(%x = %pa : !fabric.bits<32>,
                   %y = %pb : !fabric.bits<32>) -> !fabric.bits<32>
        attributes {valid_encodings = [{outputs = [0 : i32], resources = [
          {resource = 0 : i32, mode = 0 : i32}
        ]}]} {
      // expected-error @+1 {{hw_params mode #0 operation @llvm.arm.sadd16 is not a registered MLIR operation and cannot be materialized}}
      %v = fabric.op [@llvm.arm.sadd16] (%x, %y)
           {hw_params = [{op = @llvm.arm.sadd16,
             function_type = (i32, i32) -> i32,
             input_ports = [0 : i32, 1 : i32],
             output_ports = [0 : i32], attributes = {}}]}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %v : !fabric.bits<32>
    }
  }
  fabric.yield
}
