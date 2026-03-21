module {
  fabric.spatial_pe @add_pe(%lhs: !fabric.bits<64>, %rhs: !fabric.bits<64>)
      -> (!fabric.bits<64>) {
    fabric.function_unit @fu_add(%arg0: i32, %arg1: i32) -> (i32)
        [latency = 1, interval = 1] {
      %sum = arith.addi %arg0, %arg1 : i32
      fabric.yield %sum : i32
    }
    fabric.yield
  }

  fabric.module @mapper_fifo_bufferization_test(
      %a: !fabric.bits<64>, %b: !fabric.bits<64>, %c: !fabric.bits<64>)
      -> (!fabric.bits<64>) {
    %pe0 = fabric.instance @add_pe(%a, %b) {sym_name = "pe_0"}
        : (!fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>)
    %fifo = fabric.fifo @fifo_0 [depth = 4, bypassable] (%pe0#0)
        attributes {bypassed = true}
        : (!fabric.bits<64>) -> (!fabric.bits<64>)
    %pe1 = fabric.instance @add_pe(%fifo, %c) {sym_name = "pe_1"}
        : (!fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>)
    fabric.yield %pe1#0 : !fabric.bits<64>
  }
}
