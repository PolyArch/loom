// RUN: not loom --split-input-file --loom-elaborate-fabric-instances \
// RUN:   --mlir-disable-threading --mlir-print-ir-after-failure -o /dev/null \
// RUN:   %s 2>&1 | FileCheck %s

fabric.module @pe_earlier(%arg : !fabric.bits<8>) -> (!fabric.bits<8>) {
  fabric.switch @IDENTITY [spatial]
      (!fabric.bits<8>) -> (!fabric.bits<8>)
      [{connectivity_table = ["1"]}]
  %result = fabric.instantiate @IDENTITY(
      %arg : !fabric.bits<8>) -> (!fabric.bits<8>)
  fabric.yield %result : !fabric.bits<8>
}

fabric.module @pe_host(%arg : !fabric.bits_tag<16, 4>) -> () {
  fabric.pe @TEMP [temporal] (!fabric.bits_tag<8, 2>)
      -> (!fabric.bits_tag<8, 2>)
      attributes {
        tag_width = 2 : i32,
        num_instruction = 1 : i32,
        fu_config_mode = "per_fu_config",
        operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>,
        operand_buffer_size = 2 : i32
      } {
  ^bb0(%pe_arg : !fabric.bits<8>):
    fabric.fu(%fu_arg = %pe_arg : !fabric.bits<8>) -> (!fabric.bits<8>) {
      %sum = fabric.op [@arith.addi] (%fu_arg, %fu_arg)
          : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %sum : !fabric.bits<8>
    }
    fabric.yield
  }
  // CHECK: error: 'fabric.pe' op requires uniform 'bits_tag<W, T>' on all PE ports
  %unused = fabric.instantiate @TEMP(
      %arg : !fabric.bits_tag<16, 4> to !fabric.bits_tag<8, 2>)
      -> (!fabric.bits_tag<8, 2>)
  fabric.yield
}

// CHECK: IR Dump After{{.*}}ElaborateInstancesPass Failed
// CHECK-LABEL: fabric.module @pe_earlier
// CHECK: fabric.instantiate @IDENTITY
// CHECK-LABEL: fabric.module @pe_host
// CHECK: fabric.instantiate @TEMP

// -----

fabric.module @fu_earlier(%arg : !fabric.bits<8>) -> (!fabric.bits<8>) {
  fabric.switch @IDENTITY [spatial]
      (!fabric.bits<8>) -> (!fabric.bits<8>)
      [{connectivity_table = ["1"]}]
  %result = fabric.instantiate @IDENTITY(
      %arg : !fabric.bits<8>) -> (!fabric.bits<8>)
  fabric.yield %result : !fabric.bits<8>
}

fabric.module @fu_host(%arg : !fabric.bits_tag<8, 2>)
    -> (!fabric.bits_tag<8, 2>) {
  %pe = fabric.pe [temporal] (
      %pe_arg = %arg : !fabric.bits_tag<8, 2> to !fabric.bits<4>)
      -> !fabric.bits_tag<8, 2>
      attributes {
        tag_width = 2 : i32,
        num_instruction = 1 : i32,
        fu_config_mode = "per_fu_config",
        operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>,
        operand_buffer_size = 2 : i32
      } {
    fabric.fu @FU (!fabric.bits<8>) -> (!fabric.bits<8>) {
    ^bb0(%fu_arg : !fabric.bits<8>):
      %sum = fabric.op [@arith.addi] (%fu_arg, %fu_arg)
          : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %sum : !fabric.bits<8>
    }
    // CHECK: error: 'fabric.fu' op operand #0 bits-width 4 is less than block-argument bits-width 8
    %unused = fabric.instantiate @FU(
        %pe_arg : !fabric.bits<4> to !fabric.bits<8>) -> (!fabric.bits<8>)
  }
  fabric.yield %pe : !fabric.bits_tag<8, 2>
}

// CHECK: IR Dump After{{.*}}ElaborateInstancesPass Failed
// CHECK-LABEL: fabric.module @fu_earlier
// CHECK: fabric.instantiate @IDENTITY
// CHECK-LABEL: fabric.module @fu_host
// CHECK: fabric.instantiate @FU
