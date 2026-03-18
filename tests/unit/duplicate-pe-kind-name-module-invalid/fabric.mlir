module {
  fabric.module @duplicate_pe_kind_name_module_invalid(
      %a: i32, %b: i32, %c: i32) -> (i32) {
    fabric.spatial_pe @dup(%x: !fabric.bits<64>, %y: !fabric.bits<64>)
        -> (!fabric.bits<64>) {
      fabric.function_unit @fu_add(%lhs: i32, %rhs: i32) -> (i32)
          [latency = 1, interval = 1] {
        %sum = arith.addi %lhs, %rhs : i32
        fabric.yield %sum : i32
      }
      fabric.yield
    }

    fabric.temporal_pe @dup(
        %p0: !fabric.tagged<!fabric.bits<32>, i1>,
        %p1: !fabric.tagged<!fabric.bits<32>, i1>,
        %p2: !fabric.tagged<!fabric.bits<32>, i1>)
        -> (!fabric.tagged<!fabric.bits<32>, i1>)
        [num_register = 0 : i64, num_instruction = 1 : i64,
         reg_fifo_depth = 0 : i64] {
      fabric.function_unit @fu_add_t(%x: i32, %y: i32, %z: i32) -> (i32)
          [latency = 1, interval = 1] {
        %sum = arith.addi %x, %y : i32
        fabric.yield %sum : i32
      }
      fabric.yield
    }

    %r = arith.addi %a, %b : i32
    fabric.yield %r : i32
  }
}
