module {
  fabric.spatial_pe @test_pe(%p0: !fabric.bits<32>, %p1: !fabric.bits<32>, %p2: !fabric.bits<32>) -> (!fabric.bits<32>) {
    fabric.function_unit @fu_add(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.function_unit @fu_mac(%arg0: i32, %arg1: i32, %arg2: i32) -> (i32) [latency = 1, interval = 1] {
      %d = arith.muli %arg0, %arg1 : i32
      %e = arith.addi %d, %arg2 : i32
      %g = fabric.mux %d, %e {sel = 0 : i64, discard = false, disconnect = false} : i32, i32 -> i32
      fabric.yield %g : i32
    }
    fabric.yield
  }
  fabric.module @single_pe_test(%in0: !fabric.bits<32>, %in1: !fabric.bits<32>, %in2: !fabric.bits<32>, %in3: !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
    %pe = fabric.instance @test_pe(%in0, %in1, %in2) {sym_name = "pe_0"} : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>)
    fabric.yield %pe, %in3 : !fabric.bits<32>, !fabric.bits<32>
  }
}
