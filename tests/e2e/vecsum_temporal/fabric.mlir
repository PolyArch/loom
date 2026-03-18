module {
  fabric.temporal_pe @tpe_vecsum(
      %p0: !fabric.tagged<!fabric.bits<64>, i1>,
      %p1: !fabric.tagged<!fabric.bits<64>, i1>,
      %p2: !fabric.tagged<!fabric.bits<64>, i1>,
      %p3: !fabric.tagged<!fabric.bits<64>, i1>,
      %p4: !fabric.tagged<!fabric.bits<64>, i1>)
      -> (!fabric.tagged<!fabric.bits<64>, i1>,
          !fabric.tagged<!fabric.bits<64>, i1>,
          !fabric.tagged<!fabric.bits<64>, i1>)
      [
        num_register = 0 : i64,
        num_instruction = 16 : i64,
        reg_fifo_depth = 0 : i64
      ] {
    fabric.function_unit @fu_join(%a: none) -> (none)
        [latency = 1, interval = 1] {
      %0 = handshake.join %a : none
      fabric.yield %0 : none
    }

    fabric.function_unit @fu_const_index(%ctrl: none) -> (index)
        [latency = 1, interval = 1] {
      %0 = handshake.constant %ctrl {value = 0 : index} : index
      fabric.yield %0 : index
    }

    fabric.function_unit @fu_const_index_1(%ctrl: none) -> (index)
        [latency = 1, interval = 1] {
      %0 = handshake.constant %ctrl {value = 0 : index} : index
      fabric.yield %0 : index
    }

    fabric.function_unit @fu_index_cast(%arg0: i32) -> (index)
        [latency = 1, interval = 1] {
      %0 = arith.index_cast %arg0 : i32 to index
      fabric.yield %0 : index
    }

    fabric.function_unit @fu_stream(%start: index, %step: index, %bound: index)
        -> (index, i1) [latency = 1, interval = 1] {
      %0, %1 = dataflow.stream %start, %step, %bound
          {step_op = "+=", cont_cond = "<"}
          : (index, index, index) -> (index, i1)
      fabric.yield %0, %1 : index, i1
    }

    fabric.function_unit @fu_gate_index(%value: index, %cond: i1)
        -> (index, i1) [latency = 1, interval = 1] {
      %0, %1 = dataflow.gate %value, %cond : index, i1 -> index, i1
      fabric.yield %0, %1 : index, i1
    }

    fabric.function_unit @fu_gate_i32(%value: i32, %cond: i1)
        -> (i32, i1) [latency = 1, interval = 1] {
      %0, %1 = dataflow.gate %value, %cond : i32, i1 -> i32, i1
      fabric.yield %0, %1 : i32, i1
    }

    fabric.function_unit @fu_carry_i32(%cond: i1, %a: i32, %b: i32)
        -> (i32) [latency = 1, interval = 1] {
      %0 = dataflow.carry %cond, %a, %b : i1, i32, i32 -> i32
      fabric.yield %0 : i32
    }

    fabric.function_unit @fu_carry_none(%cond: i1, %a: none, %b: none)
        -> (none) [latency = 1, interval = 1] {
      %0 = dataflow.carry %cond, %a, %b : i1, none, none -> none
      fabric.yield %0 : none
    }

    fabric.function_unit @fu_cond_br_i32(%cond: i1, %value: i32)
        -> (i32, i32) [latency = 1, interval = 1] {
      %0, %1 = handshake.cond_br %cond, %value : i32
      fabric.yield %0, %1 : i32, i32
    }

    fabric.function_unit @fu_cond_br_none(%cond: i1, %value: none)
        -> (none, none) [latency = 1, interval = 1] {
      %0, %1 = handshake.cond_br %cond, %value : none
      fabric.yield %0, %1 : none, none
    }

    fabric.function_unit @fu_load(%addr: index, %data: i32, %ctrl: none)
        -> (i32, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.load [%addr] %data, %ctrl : index, i32
      fabric.yield %0, %1 : i32, index
    }

    fabric.function_unit @fu_addi(%a: i32, %b: i32) -> (i32)
        [latency = 1, interval = 1] {
      %0 = arith.addi %a, %b : i32
      fabric.yield %0 : i32
    }

    fabric.yield
  }

  fabric.module @vecsum_temporal_domain(
      %mem0: memref<?xi32>,
      %n: !fabric.bits<64>,
      %init: !fabric.bits<64>,
      %ctrl: !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>) {
    %ext0:2 = fabric.extmemory @extmem_0
        [ldCount = 1, stCount = 0, lsqDepth = 0, memrefType = memref<?xi32>]
        (%mem0, %addr_bits)
        : (memref<?xi32>, !fabric.bits<64>)
          -> (!fabric.bits<64>, !fabric.bits<64>)

    %tag_n = fabric.add_tag %n {tag = 0 : i64}
        : !fabric.bits<64> -> !fabric.tagged<!fabric.bits<64>, i1>
    %tag_init = fabric.add_tag %init {tag = 0 : i64}
        : !fabric.bits<64> -> !fabric.tagged<!fabric.bits<64>, i1>
    %tag_ctrl = fabric.add_tag %ctrl {tag = 0 : i64}
        : !fabric.bits<64> -> !fabric.tagged<!fabric.bits<64>, i1>
    %tag_lddata = fabric.add_tag %ext0#0 {tag = 0 : i64}
        : !fabric.bits<64> -> !fabric.tagged<!fabric.bits<64>, i1>
    %tag_lddone = fabric.add_tag %ext0#1 {tag = 0 : i64}
        : !fabric.bits<64> -> !fabric.tagged<!fabric.bits<64>, i1>

    %tpe0:3 = fabric.instance @tpe_vecsum(
        %tag_n, %tag_init, %tag_ctrl, %tag_lddata, %tag_lddone)
        {sym_name = "tpe_0"}
        : (!fabric.tagged<!fabric.bits<64>, i1>,
           !fabric.tagged<!fabric.bits<64>, i1>,
           !fabric.tagged<!fabric.bits<64>, i1>,
           !fabric.tagged<!fabric.bits<64>, i1>,
           !fabric.tagged<!fabric.bits<64>, i1>)
          -> (!fabric.tagged<!fabric.bits<64>, i1>,
              !fabric.tagged<!fabric.bits<64>, i1>,
              !fabric.tagged<!fabric.bits<64>, i1>)

    %addr_bits = fabric.del_tag %tpe0#0
        : !fabric.tagged<!fabric.bits<64>, i1> -> !fabric.bits<64>
    %sum_bits = fabric.del_tag %tpe0#1
        : !fabric.tagged<!fabric.bits<64>, i1> -> !fabric.bits<64>
    %done_bits = fabric.del_tag %tpe0#2
        : !fabric.tagged<!fabric.bits<64>, i1> -> !fabric.bits<64>

    fabric.yield %sum_bits, %done_bits : !fabric.bits<64>, !fabric.bits<64>
  }
}
