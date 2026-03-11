module {
  fabric.pe @pe_addi_i32_i32(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.addi %arg0, %arg1 : i32
    fabric.yield %0 : i32
  }
  fabric.pe @pe_andi_i32_i32(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.andi %arg0, %arg1 : i32
    fabric.yield %0 : i32
  }
  fabric.pe @pe_cmpi_i32_i32_to_o1(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<1>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.cmpi eq, %arg0, %arg1 : i32
    fabric.yield %0 : i1
  }
  fabric.pe @pe_extui_i32_to_o64(%arg0: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<64>) {
  ^bb0(%arg0: i32):
    %0 = arith.extui %arg0 : i32 to i64
    fabric.yield %0 : i64
  }
  fabric.pe @pe_index_cast_i32_to_o57(%arg0: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<57>) {
  ^bb0(%arg0: i32):
    %0 = arith.index_cast %arg0 : i32 to index
    fabric.yield %0 : index
  }
  fabric.pe @pe_index_cast_i57_to_o32(%arg0: !dataflow.bits<57>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>) {
  ^bb0(%arg0: index):
    %0 = arith.index_cast %arg0 : index to i32
    fabric.yield %0 : i32
  }
  fabric.pe @pe_index_cast_i64_to_o57(%arg0: !dataflow.bits<64>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<57>) {
  ^bb0(%arg0: i64):
    %0 = arith.index_cast %arg0 : i64 to index
    fabric.yield %0 : index
  }
  fabric.pe @pe_select_i1_i57_i57(%arg0: !dataflow.bits<1>, %arg1: !dataflow.bits<57>, %arg2: !dataflow.bits<57>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<57>) {
  ^bb0(%arg0: i1, %arg1: index, %arg2: index):
    %0 = arith.select %arg0, %arg1, %arg2 : index
    fabric.yield %0 : index
  }
  fabric.pe @pe_shli_i32_i32(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.shli %arg0, %arg1 : i32
    fabric.yield %0 : i32
  }
  fabric.pe @pe_shrui_i32_i32(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.shrui %arg0, %arg1 : i32
    fabric.yield %0 : i32
  }
  fabric.pe @pe_xori_i32_i32(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.xori %arg0, %arg1 : i32
    fabric.yield %0 : i32
  }
  fabric.pe @pe_carry_i1_i0_i0_to_o0(%arg0: !dataflow.bits<1>, %arg1: none, %arg2: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
  ^bb0(%arg0: i1, %arg1: none, %arg2: none):
    %0 = dataflow.carry %arg0, %arg1, %arg2 : i1, none, none -> none
    fabric.yield %0 : none
  }
  fabric.pe @pe_carry_i1_i32_i32(%arg0: !dataflow.bits<1>, %arg1: !dataflow.bits<32>, %arg2: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i1, %arg1: i32, %arg2: i32):
    %0 = dataflow.carry %arg0, %arg1, %arg2 : i1, i32, i32 -> i32
    fabric.yield %0 : i32
  }
  fabric.pe @pe_gate_i32_i1_to_o32_o1(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<1>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>, !dataflow.bits<1>) {
  ^bb0(%arg0: i32, %arg1: i1):
    %afterValue, %afterCond = dataflow.gate %arg0, %arg1 : i32, i1 -> i32, i1
    fabric.yield %afterValue, %afterCond : i32, i1
  }
  fabric.pe @pe_gate_i57_i1_to_o57_o1(%arg0: !dataflow.bits<57>, %arg1: !dataflow.bits<1>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<57>, !dataflow.bits<1>) {
  ^bb0(%arg0: index, %arg1: i1):
    %afterValue, %afterCond = dataflow.gate %arg0, %arg1 : index, i1 -> index, i1
    fabric.yield %afterValue, %afterCond : index, i1
  }
  fabric.pe @pe_invariant_i1_i0_to_o0(%arg0: !dataflow.bits<1>, %arg1: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
  ^bb0(%arg0: i1, %arg1: none):
    %0 = dataflow.invariant %arg0, %arg1 : i1, none -> none
    fabric.yield %0 : none
  }
  fabric.pe @pe_invariant_i1_i32(%arg0: !dataflow.bits<1>, %arg1: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i1, %arg1: i32):
    %0 = dataflow.invariant %arg0, %arg1 : i1, i32 -> i32
    fabric.yield %0 : i32
  }
  fabric.pe @pe_stream_i57_i57_i57_to_o57_o1(%arg0: !dataflow.bits<57>, %arg1: !dataflow.bits<57>, %arg2: !dataflow.bits<57>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<57>, !dataflow.bits<1>) {
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    %index, %willContinue = dataflow.stream %arg0, %arg1, %arg2
    fabric.yield %index, %willContinue : index, i1
  }
  fabric.pe @pe_cond_br_i1_i0_to_o0_o0(%arg0: !dataflow.bits<1>, %arg1: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none, none) {
  ^bb0(%arg0: i1, %arg1: none):
    %trueResult, %falseResult = handshake.cond_br %arg0, %arg1 : none
    fabric.yield %trueResult, %falseResult : none, none
  }
  fabric.pe @pe_cond_br_i1_i32(%arg0: !dataflow.bits<1>, %arg1: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>, !dataflow.bits<32>) {
  ^bb0(%arg0: i1, %arg1: i32):
    %trueResult, %falseResult = handshake.cond_br %arg0, %arg1 : i32
    fabric.yield %trueResult, %falseResult : i32, i32
  }
  fabric.pe @pe_constant_i0_to_o32(%arg0: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>) {
  ^bb0(%arg0: none):
    %0 = handshake.constant %arg0 {value = 0 : i32} : i32
    fabric.yield %0 : i32
  }
  fabric.pe @pe_constant_i0_to_o57(%arg0: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<57>) {
  ^bb0(%arg0: none):
    %0 = handshake.constant %arg0 {value = 0 : index} : index
    fabric.yield %0 : index
  }
  fabric.pe @pe_constant_i0_to_o64(%arg0: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<64>) {
  ^bb0(%arg0: none):
    %0 = handshake.constant %arg0 {value = 0 : i64} : i64
    fabric.yield %0 : i64
  }
  fabric.pe @pe_join_i0(%arg0: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
    %0 = handshake.join %arg0 : none
    fabric.yield %0 : none
  }
  fabric.pe @pe_join_i0_i0(%arg0: none, %arg1: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
    %0 = handshake.join %arg0, %arg1 : none, none
    fabric.yield %0 : none
  }
  fabric.pe @pe_join_i0_i0_i0(%arg0: none, %arg1: none, %arg2: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
    %0 = handshake.join %arg0, %arg1, %arg2 : none, none, none
    fabric.yield %0 : none
  }
  fabric.pe @pe_mux_i57_i0_i0_to_o0(%arg0: !dataflow.bits<57>, %arg1: none, %arg2: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
  ^bb0(%arg0: index, %arg1: none, %arg2: none):
    %0 = handshake.mux %arg0 [%arg1, %arg2] : index, none
    fabric.yield %0 : none
  }
  fabric.pe @pe_mux_i57_i32_i32_to_o32(%arg0: !dataflow.bits<57>, %arg1: !dataflow.bits<32>, %arg2: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>) {
  ^bb0(%arg0: index, %arg1: i32, %arg2: i32):
    %0 = handshake.mux %arg0 [%arg1, %arg2] : index, i32
    fabric.yield %0 : i32
  }
  fabric.pe @pe_sink_i1(%arg0: !dataflow.bits<1>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> () {
  ^bb0(%arg0: i1):
    handshake.sink %arg0 : i1
    fabric.yield
  }
  fabric.pe @load_pe_w32(%arg0: !dataflow.bits<57>, %arg1: !dataflow.bits<32>, %arg2: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>, !dataflow.bits<57>) {
  ^bb0(%arg0: index, %arg1: i32, %arg2: none):
    %dataResult, %addressResults = handshake.load [%arg0] %arg1, %arg2 : index, i32
    fabric.yield %dataResult, %addressResults : i32, index
  }
  fabric.pe @store_pe_w32(%arg0: !dataflow.bits<57>, %arg1: !dataflow.bits<32>, %arg2: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>, !dataflow.bits<57>) {
  ^bb0(%arg0: index, %arg1: i32, %arg2: none):
    %dataResult, %addressResult = handshake.store [%arg0] %arg1, %arg2 : index, i32
    fabric.yield %dataResult, %addressResult : i32, index
  }
  fabric.module @genadg_2(%arg0: memref<?xi32, strided<[1], offset: ?>>, %arg1: memref<?xi32, strided<[1], offset: ?>>, %arg2: none, %arg3: !dataflow.bits<32>) -> (none) {
    %0:5 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111"]] %1#4, %1#5, %7#0, %7#1, %arg2, %239, %240#1, %241#1 : none -> none, none, none, none, none
    %1:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %2#4, %2#5, %8#0, %8#1, %0#0, %0#1 : none -> none, none, none, none, none, none, none
    %2:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %3#4, %3#5, %9#0, %9#1, %1#0, %1#1 : none -> none, none, none, none, none, none, none
    %3:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %4#4, %4#5, %10#0, %10#1, %2#0, %2#1 : none -> none, none, none, none, none, none, none
    %4:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %5#4, %5#5, %11#0, %11#1, %3#0, %3#1 : none -> none, none, none, none, none, none, none
    %5:7 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %6#2, %6#3, %12#0, %12#1, %4#0, %4#1, %211#1 : none -> none, none, none, none, none, none, none
    %6:4 = fabric.switch [connectivity_table = ["11111", "11111", "11111", "11111"]] %13#0, %13#1, %5#0, %5#1, %212#1 : none -> none, none, none, none
    %7:8 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %0#2, %0#3, %8#6, %8#7, %14#0, %14#1 : none -> none, none, none, none, none, none, none, none
    %8:10 = fabric.switch [connectivity_table = ["1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111"]] %1#2, %1#3, %9#6, %9#7, %15#0, %15#1, %7#2, %7#3, %198, %213#1 : none -> none, none, none, none, none, none, none, none, none, none
    %9:10 = fabric.switch [connectivity_table = ["1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111"]] %2#2, %2#3, %10#6, %10#7, %16#0, %16#1, %8#2, %8#3, %199, %214#1 : none -> none, none, none, none, none, none, none, none, none, none
    %10:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %3#2, %3#3, %11#6, %11#7, %17#0, %17#1, %9#2, %9#3, %200 : none -> none, none, none, none, none, none, none, none, none
    %11:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %4#2, %4#3, %12#6, %12#7, %18#0, %18#1, %10#2, %10#3, %207 : none -> none, none, none, none, none, none, none, none, none
    %12:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %5#2, %5#3, %13#4, %13#5, %19#0, %19#1, %11#2, %11#3, %211#0 : none -> none, none, none, none, none, none, none, none, none
    %13:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %6#0, %6#1, %20#0, %20#1, %12#2, %12#3, %212#0 : none -> none, none, none, none, none, none
    %14:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %7#4, %7#5, %15#6, %15#7, %21#0, %21#1 : none -> none, none, none, none, none, none, none
    %15:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %8#4, %8#5, %16#6, %16#7, %22#0, %22#1, %14#2, %14#3, %213#0 : none -> none, none, none, none, none, none, none, none, none
    %16:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %9#4, %9#5, %17#6, %17#7, %23#0, %23#1, %15#2, %15#3, %214#0 : none -> none, none, none, none, none, none, none, none, none
    %17:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %10#4, %10#5, %18#6, %18#7, %24#0, %24#1, %16#2, %16#3 : none -> none, none, none, none, none, none, none, none, none
    %18:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %11#4, %11#5, %19#6, %19#7, %25#0, %25#1, %17#2, %17#3 : none -> none, none, none, none, none, none, none, none, none
    %19:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %12#4, %12#5, %20#4, %20#5, %26#0, %26#1, %18#2, %18#3 : none -> none, none, none, none, none, none, none, none, none
    %20:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %13#2, %13#3, %27#0, %27#1, %19#2, %19#3 : none -> none, none, none, none, none, none
    %21:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %14#4, %14#5, %22#6, %22#7, %28#0, %28#1 : none -> none, none, none, none, none, none, none
    %22:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %15#4, %15#5, %23#6, %23#7, %29#0, %29#1, %21#2, %21#3 : none -> none, none, none, none, none, none, none, none, none
    %23:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %16#4, %16#5, %24#6, %24#7, %30#0, %30#1, %22#2, %22#3 : none -> none, none, none, none, none, none, none, none, none
    %24:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %17#4, %17#5, %25#6, %25#7, %31#0, %31#1, %23#2, %23#3 : none -> none, none, none, none, none, none, none, none, none
    %25:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %18#4, %18#5, %26#6, %26#7, %32#0, %32#1, %24#2, %24#3 : none -> none, none, none, none, none, none, none, none, none
    %26:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %19#4, %19#5, %27#4, %27#5, %33#0, %33#1, %25#2, %25#3 : none -> none, none, none, none, none, none, none, none, none
    %27:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %20#2, %20#3, %34#0, %34#1, %26#2, %26#3 : none -> none, none, none, none, none, none, none
    %28:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %21#4, %21#5, %29#6, %29#7, %35#0, %35#1 : none -> none, none, none, none, none, none, none
    %29:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %22#4, %22#5, %30#6, %30#7, %36#0, %36#1, %28#2, %28#3 : none -> none, none, none, none, none, none, none, none, none
    %30:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %23#4, %23#5, %31#6, %31#7, %37#0, %37#1, %29#2, %29#3 : none -> none, none, none, none, none, none, none, none, none
    %31:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %24#4, %24#5, %32#6, %32#7, %38#0, %38#1, %30#2, %30#3 : none -> none, none, none, none, none, none, none, none, none
    %32:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %25#4, %25#5, %33#6, %33#7, %39#0, %39#1, %31#2, %31#3, %230 : none -> none, none, none, none, none, none, none, none, none, none
    %33:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %26#4, %26#5, %34#4, %34#5, %40#0, %40#1, %32#2, %32#3, %231 : none -> none, none, none, none, none, none, none, none, none
    %34:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %27#2, %27#3, %41#0, %41#1, %33#2, %33#3, %232 : none -> none, none, none, none, none, none
    %35:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %28#4, %28#5, %36#6, %36#7, %42#0, %42#1 : none -> none, none, none, none, none, none, none
    %36:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %29#4, %29#5, %37#6, %37#7, %43#0, %43#1, %35#2, %35#3, %233 : none -> none, none, none, none, none, none, none, none, none
    %37:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %30#4, %30#5, %38#6, %38#7, %44#0, %44#1, %36#2, %36#3, %234 : none -> none, none, none, none, none, none, none, none
    %38:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %31#4, %31#5, %39#6, %39#7, %45#0, %45#1, %37#2, %37#3 : none -> none, none, none, none, none, none, none, none
    %39:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %32#4, %32#5, %40#6, %40#7, %46#0, %46#1, %38#2, %38#3 : none -> none, none, none, none, none, none, none, none
    %40:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %33#4, %33#5, %41#4, %41#5, %47#0, %47#1, %39#2, %39#3 : none -> none, none, none, none, none, none, none, none
    %41:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %34#2, %34#3, %48#0, %48#1, %40#2, %40#3 : none -> none, none, none, none, none, none
    %42:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %35#4, %35#5, %43#4, %43#5 : none -> none, none, none, none
    %43:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %36#4, %36#5, %44#4, %44#5, %42#2, %42#3 : none -> none, none, none, none, none, none
    %44:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %37#4, %37#5, %45#4, %45#5, %43#2, %43#3 : none -> none, none, none, none, none, none
    %45:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %38#4, %38#5, %46#4, %46#5, %44#2, %44#3 : none -> none, none, none, none, none, none
    %46:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %39#4, %39#5, %47#4, %47#5, %45#2, %45#3 : none -> none, none, none, none, none, none
    %47:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %40#4, %40#5, %48#2, %48#3, %46#2, %46#3 : none -> none, none, none, none, none, none
    %48:5 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111"]] %41#2, %41#3, %47#2, %47#3 : none -> none, none, none, none, none
    %49:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %50#4, %50#5, %55#0, %55#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %50:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %51#4, %51#5, %56#0, %56#1, %49#0, %49#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %51:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %52#4, %52#5, %57#0, %57#1, %50#0, %50#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %52:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %53#4, %53#5, %58#0, %58#1, %51#0, %51#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %53:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %54#2, %54#3, %59#0, %59#1, %52#0, %52#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %54:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %60#0, %60#1, %53#0, %53#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %55:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %49#2, %49#3, %56#6, %56#7, %61#0, %61#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %56:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %50#2, %50#3, %57#6, %57#7, %62#0, %62#1, %55#2, %55#3, %181 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %57:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %51#2, %51#3, %58#6, %58#7, %63#0, %63#1, %56#2, %56#3, %182 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %58:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %52#2, %52#3, %59#6, %59#7, %64#0, %64#1, %57#2, %57#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %59:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %53#2, %53#3, %60#4, %60#5, %65#0, %65#1, %58#2, %58#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %60:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %54#0, %54#1, %66#0, %66#1, %59#2, %59#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %61:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %55#4, %55#5, %62#6, %62#7, %67#0, %67#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %62:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %56#4, %56#5, %63#6, %63#7, %68#0, %68#1, %61#2, %61#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %63:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %57#4, %57#5, %64#6, %64#7, %69#0, %69#1, %62#2, %62#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %64:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %58#4, %58#5, %65#6, %65#7, %70#0, %70#1, %63#2, %63#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %65:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %59#4, %59#5, %66#4, %66#5, %71#0, %71#1, %64#2, %64#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %66:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %60#2, %60#3, %72#0, %72#1, %65#2, %65#3, %203#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %67:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %61#4, %61#5, %68#6, %68#7, %73#0, %73#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %68:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %62#4, %62#5, %69#6, %69#7, %74#0, %74#1, %67#2, %67#3, %204#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %69:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %63#4, %63#5, %70#6, %70#7, %75#0, %75#1, %68#2, %68#3, %205#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %70:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %64#4, %64#5, %71#6, %71#7, %76#0, %76#1, %69#2, %69#3, %206#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %71:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %65#4, %65#5, %72#4, %72#5, %77#0, %77#1, %70#2, %70#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %72:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %66#2, %66#3, %78#0, %78#1, %71#2, %71#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %73:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %67#4, %67#5, %74#6, %74#7, %79#0, %79#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %74:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %68#4, %68#5, %75#6, %75#7, %80#0, %80#1, %73#2, %73#3, %209#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %75:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %69#4, %69#5, %76#6, %76#7, %81#0, %81#1, %74#2, %74#3, %210#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %76:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %70#4, %70#5, %77#6, %77#7, %82#0, %82#1, %75#2, %75#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %77:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %71#4, %71#5, %78#4, %78#5, %83#0, %83#1, %76#2, %76#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %78:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %72#2, %72#3, %84#0, %84#1, %77#2, %77#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %79:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %73#4, %73#5, %80#4, %80#5 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %80:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %74#4, %74#5, %81#4, %81#5, %79#2, %79#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %81:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %75#4, %75#5, %82#4, %82#5, %80#2, %80#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %82:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %76#4, %76#5, %83#4, %83#5, %81#2, %81#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %83:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %77#4, %77#5, %84#2, %84#3, %82#2, %82#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %84:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %78#2, %78#3, %83#2, %83#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %85:5 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111"]] %86#4, %86#5, %92#0, %92#1, %arg3, %240#0, %241#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %86:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %87#4, %87#5, %93#0, %93#1, %85#0, %85#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %87:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %88#4, %88#5, %94#0, %94#1, %86#0, %86#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %88:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %89#4, %89#5, %95#0, %95#1, %87#0, %87#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %89:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %90#4, %90#5, %96#0, %96#1, %88#0, %88#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %90:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %91#2, %91#3, %97#0, %97#1, %89#0, %89#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %91:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %98#0, %98#1, %90#0, %90#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %92:8 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %85#2, %85#3, %93#6, %93#7, %99#0, %99#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %93:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %86#2, %86#3, %94#6, %94#7, %100#0, %100#1, %92#2, %92#3, %179 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %94:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %87#2, %87#3, %95#6, %95#7, %101#0, %101#1, %93#2, %93#3, %180 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %95:10 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %88#2, %88#3, %96#6, %96#7, %102#0, %102#1, %94#2, %94#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %96:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %89#2, %89#3, %97#6, %97#7, %103#0, %103#1, %95#2, %95#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %97:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %90#2, %90#3, %98#4, %98#5, %104#0, %104#1, %96#2, %96#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %98:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %91#0, %91#1, %105#0, %105#1, %97#2, %97#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %99:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %92#4, %92#5, %100#6, %100#7, %106#0, %106#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %100:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %93#4, %93#5, %101#6, %101#7, %107#0, %107#1, %99#2, %99#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %101:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %94#4, %94#5, %102#6, %102#7, %108#0, %108#1, %100#2, %100#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %102:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %95#4, %95#5, %103#6, %103#7, %109#0, %109#1, %101#2, %101#3, %187 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %103:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %96#4, %96#5, %104#6, %104#7, %110#0, %110#1, %102#2, %102#3, %192 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %104:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %97#4, %97#5, %105#4, %105#5, %111#0, %111#1, %103#2, %103#3, %193 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %105:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %98#2, %98#3, %112#0, %112#1, %104#2, %104#3, %194 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %106:8 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %99#4, %99#5, %107#6, %107#7, %113#0, %113#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %107:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %100#4, %100#5, %108#6, %108#7, %114#0, %114#1, %106#2, %106#3, %195 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %108:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %101#4, %101#5, %109#6, %109#7, %115#0, %115#1, %107#2, %107#3, %196 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %109:10 = fabric.switch [connectivity_table = ["1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111"]] %102#4, %102#5, %110#6, %110#7, %116#0, %116#1, %108#2, %108#3, %197, %215#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %110:9 = fabric.switch [connectivity_table = ["1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111"]] %103#4, %103#5, %111#6, %111#7, %117#0, %117#1, %109#2, %109#3, %201, %216#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %111:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %104#4, %104#5, %112#4, %112#5, %118#0, %118#1, %110#2, %110#3, %202 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %112:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %105#2, %105#3, %119#0, %119#1, %111#2, %111#3, %203#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %113:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %106#4, %106#5, %114#6, %114#7, %120#0, %120#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %114:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %107#4, %107#5, %115#6, %115#7, %121#0, %121#1, %113#2, %113#3, %204#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %115:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %108#4, %108#5, %116#6, %116#7, %122#0, %122#1, %114#2, %114#3, %208 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %116:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %109#4, %109#5, %117#6, %117#7, %123#0, %123#1, %115#2, %115#3, %215#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %117:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %110#4, %110#5, %118#6, %118#7, %124#0, %124#1, %116#2, %116#3, %216#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %118:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %111#4, %111#5, %119#4, %119#5, %125#0, %125#1, %117#2, %117#3, %217 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %119:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %112#2, %112#3, %126#0, %126#1, %118#2, %118#3, %218 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %120:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %113#4, %113#5, %121#6, %121#7, %127#0, %127#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %121:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %114#4, %114#5, %122#6, %122#7, %128#0, %128#1, %120#2, %120#3, %219 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %122:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %115#4, %115#5, %123#6, %123#7, %129#0, %129#1, %121#2, %121#3, %220 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %123:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %116#4, %116#5, %124#6, %124#7, %130#0, %130#1, %122#2, %122#3, %221 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %124:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %117#4, %117#5, %125#6, %125#7, %131#0, %131#1, %123#2, %123#3, %222 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %125:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %118#4, %118#5, %126#4, %126#5, %132#0, %132#1, %124#2, %124#3, %235 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %126:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %119#2, %119#3, %133#0, %133#1, %125#2, %125#3, %236#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %127:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %120#4, %120#5, %128#4, %128#5 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %128:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %121#4, %121#5, %129#4, %129#5, %127#2, %127#3, %237#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %129:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %122#4, %122#5, %130#4, %130#5, %128#2, %128#3, %238#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %130:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %123#4, %123#5, %131#4, %131#5, %129#2, %129#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %131:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %124#4, %124#5, %132#4, %132#5, %130#2, %130#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %132:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %125#4, %125#5, %133#2, %133#3, %131#2, %131#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %133:5 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111"]] %126#2, %126#3, %132#2, %132#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %134:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %135#4, %135#5, %140#0, %140#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %135:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %136#4, %136#5, %141#0, %141#1, %134#0, %134#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %136:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %137#4, %137#5, %142#0, %142#1, %135#0, %135#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %137:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %138#4, %138#5, %143#0, %143#1, %136#0, %136#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %138:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %139#2, %139#3, %144#0, %144#1, %137#0, %137#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %139:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %145#0, %145#1, %138#0, %138#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %140:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %134#2, %134#3, %141#6, %141#7, %146#0, %146#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %141:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %135#2, %135#3, %142#6, %142#7, %147#0, %147#1, %140#2, %140#3, %184 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %142:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %136#2, %136#3, %143#6, %143#7, %148#0, %148#1, %141#2, %141#3, %185 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %143:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %137#2, %137#3, %144#6, %144#7, %149#0, %149#1, %142#2, %142#3, %186 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %144:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %138#2, %138#3, %145#4, %145#5, %150#0, %150#1, %143#2, %143#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %145:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %139#0, %139#1, %151#0, %151#1, %144#2, %144#3, %188 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %146:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %140#4, %140#5, %147#6, %147#7, %152#0, %152#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %147:11 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %141#4, %141#5, %148#6, %148#7, %153#0, %153#1, %146#2, %146#3, %189 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %148:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %142#4, %142#5, %149#6, %149#7, %154#0, %154#1, %147#2, %147#3, %190 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %149:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %143#4, %143#5, %150#6, %150#7, %155#0, %155#1, %148#2, %148#3, %191 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %150:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %144#4, %144#5, %151#4, %151#5, %156#0, %156#1, %149#2, %149#3, %205#0 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %151:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %145#2, %145#3, %157#0, %157#1, %150#2, %150#3, %206#0 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %152:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %146#4, %146#5, %153#6, %153#7, %158#0, %158#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %153:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %147#4, %147#5, %154#6, %154#7, %159#0, %159#1, %152#2, %152#3, %209#0 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %154:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %148#4, %148#5, %155#6, %155#7, %160#0, %160#1, %153#2, %153#3, %210#0 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %155:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %149#4, %149#5, %156#6, %156#7, %161#0, %161#1, %154#2, %154#3, %223 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %156:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %150#4, %150#5, %157#4, %157#5, %162#0, %162#1, %155#2, %155#3, %224 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %157:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %151#2, %151#3, %163#0, %163#1, %156#2, %156#3, %225 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %158:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %152#4, %152#5, %159#6, %159#7, %164#0, %164#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %159:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %153#4, %153#5, %160#6, %160#7, %165#0, %165#1, %158#2, %158#3, %226 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %160:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %154#4, %154#5, %161#6, %161#7, %166#0, %166#1, %159#2, %159#3, %227 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %161:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %155#4, %155#5, %162#6, %162#7, %167#0, %167#1, %160#2, %160#3, %228 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %162:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %156#4, %156#5, %163#4, %163#5, %168#0, %168#1, %161#2, %161#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %163:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %157#2, %157#3, %169#0, %169#1, %162#2, %162#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %164:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %158#4, %158#5, %165#4, %165#5 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %165:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %159#4, %159#5, %166#4, %166#5, %164#2, %164#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %166:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %160#4, %160#5, %167#4, %167#5, %165#2, %165#3, %236#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %167:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %161#4, %161#5, %168#4, %168#5, %166#2, %166#3, %237#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %168:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %162#4, %162#5, %169#2, %169#3, %167#2, %167#3, %238#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %169:7 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111", "1111", "1111"]] %163#2, %163#3, %168#2, %168#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %170:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %171#4, %171#5, %173#0, %173#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %171:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %172#2, %172#3, %174#0, %174#1, %170#0, %170#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %172:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %175#0, %175#1, %171#0, %171#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %173:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %170#2, %170#3, %174#6, %174#7, %176#0, %176#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %174:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %171#2, %171#3, %175#4, %175#5, %177#0, %177#1, %173#2, %173#3, %183 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %175:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %172#0, %172#1, %178#0, %178#1, %174#2, %174#3 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %176:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %173#4, %173#5, %177#4, %177#5 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %177:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %174#4, %174#5, %178#2, %178#3, %176#2, %176#3 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %178:4 = fabric.switch [connectivity_table = ["11111", "11111", "11111", "11111"]] %175#2, %175#3, %177#2, %177#3, %229 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %179 = fabric.instance @pe_addi_i32_i32(%85#4, %92#7) {sym_name = "pe_addi_i32_i32_r0_c0"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %180 = fabric.instance @pe_andi_i32_i32(%86#6, %93#9) {sym_name = "pe_andi_i32_i32_r0_c1"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %181 = fabric.instance @pe_cmpi_i32_i32_to_o1(%87#6, %94#8) {sym_name = "pe_cmpi_i32_i32_to_o1_r0_c2"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<1>
    %182 = fabric.instance @pe_cmpi_i32_i32_to_o1(%88#6, %95#9) {sym_name = "pe_cmpi_i32_i32_to_o1_r0_c3"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<1>
    %183 = fabric.instance @pe_extui_i32_to_o64(%89#6) {sym_name = "pe_extui_i32_to_o64_r0_c4"} : (!dataflow.bits<32>) -> !dataflow.bits<64>
    %184 = fabric.instance @pe_index_cast_i32_to_o57(%90#6) {sym_name = "pe_index_cast_i32_to_o57_r0_c5"} : (!dataflow.bits<32>) -> !dataflow.bits<57>
    %185 = fabric.instance @pe_index_cast_i32_to_o57(%92#6) {sym_name = "pe_index_cast_i32_to_o57_r1_c0"} : (!dataflow.bits<32>) -> !dataflow.bits<57>
    %186 = fabric.instance @pe_index_cast_i32_to_o57(%93#8) {sym_name = "pe_index_cast_i32_to_o57_r1_c1"} : (!dataflow.bits<32>) -> !dataflow.bits<57>
    %187 = fabric.instance @pe_index_cast_i57_to_o32(%137#6) {sym_name = "pe_index_cast_i57_to_o32_r0_c3"} : (!dataflow.bits<57>) -> !dataflow.bits<32>
    %188 = fabric.instance @pe_index_cast_i64_to_o57(%171#6) {sym_name = "pe_index_cast_i64_to_o57_r0_c1"} : (!dataflow.bits<64>) -> !dataflow.bits<57>
    %189 = fabric.instance @pe_index_cast_i64_to_o57(%173#6) {sym_name = "pe_index_cast_i64_to_o57_r1_c0"} : (!dataflow.bits<64>) -> !dataflow.bits<57>
    %190 = fabric.instance @pe_select_i1_i57_i57(%51#6, %141#8, %147#9) {sym_name = "pe_select_i1_i57_i57_r1_c1"} : (!dataflow.bits<1>, !dataflow.bits<57>, !dataflow.bits<57>) -> !dataflow.bits<57>
    %191 = fabric.instance @pe_select_i1_i57_i57(%52#6, %142#8, %148#8) {sym_name = "pe_select_i1_i57_i57_r1_c2"} : (!dataflow.bits<1>, !dataflow.bits<57>, !dataflow.bits<57>) -> !dataflow.bits<57>
    %192 = fabric.instance @pe_shli_i32_i32(%95#8, %102#9) {sym_name = "pe_shli_i32_i32_r1_c3"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %193 = fabric.instance @pe_shrui_i32_i32(%96#8, %103#9) {sym_name = "pe_shrui_i32_i32_r1_c4"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %194 = fabric.instance @pe_shrui_i32_i32(%97#8, %104#9) {sym_name = "pe_shrui_i32_i32_r1_c5"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %195 = fabric.instance @pe_xori_i32_i32(%99#6, %106#7) {sym_name = "pe_xori_i32_i32_r2_c0"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %196 = fabric.instance @pe_xori_i32_i32(%100#8, %107#9) {sym_name = "pe_xori_i32_i32_r2_c1"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %197 = fabric.instance @pe_xori_i32_i32(%101#8, %108#9) {sym_name = "pe_xori_i32_i32_r2_c2"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %198 = fabric.instance @pe_carry_i1_i0_i0_to_o0(%53#6, %0#4, %7#7) {sym_name = "pe_carry_i1_i0_i0_to_o0_r0_c4"} : (!dataflow.bits<1>, none, none) -> none
    %199 = fabric.instance @pe_carry_i1_i0_i0_to_o0(%55#6, %1#6, %8#9) {sym_name = "pe_carry_i1_i0_i0_to_o0_r1_c0"} : (!dataflow.bits<1>, none, none) -> none
    %200 = fabric.instance @pe_carry_i1_i0_i0_to_o0(%56#8, %2#6, %9#9) {sym_name = "pe_carry_i1_i0_i0_to_o0_r1_c1"} : (!dataflow.bits<1>, none, none) -> none
    %201 = fabric.instance @pe_carry_i1_i32_i32(%57#8, %102#8, %109#9) {sym_name = "pe_carry_i1_i32_i32_r2_c3"} : (!dataflow.bits<1>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %202 = fabric.instance @pe_carry_i1_i32_i32(%58#8, %103#8, %110#8) {sym_name = "pe_carry_i1_i32_i32_r2_c4"} : (!dataflow.bits<1>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %203:2 = fabric.instance @pe_gate_i32_i1_to_o32_o1(%104#8, %59#8) {sym_name = "pe_gate_i32_i1_to_o32_o1_r2_c5"} : (!dataflow.bits<32>, !dataflow.bits<1>) -> (!dataflow.bits<32>, !dataflow.bits<1>)
    %204:2 = fabric.instance @pe_gate_i32_i1_to_o32_o1(%106#6, %61#6) {sym_name = "pe_gate_i32_i1_to_o32_o1_r3_c0"} : (!dataflow.bits<32>, !dataflow.bits<1>) -> (!dataflow.bits<32>, !dataflow.bits<1>)
    %205:2 = fabric.instance @pe_gate_i57_i1_to_o57_o1(%143#8, %62#8) {sym_name = "pe_gate_i57_i1_to_o57_o1_r1_c3"} : (!dataflow.bits<57>, !dataflow.bits<1>) -> (!dataflow.bits<57>, !dataflow.bits<1>)
    %206:2 = fabric.instance @pe_gate_i57_i1_to_o57_o1(%144#8, %63#8) {sym_name = "pe_gate_i57_i1_to_o57_o1_r1_c4"} : (!dataflow.bits<57>, !dataflow.bits<1>) -> (!dataflow.bits<57>, !dataflow.bits<1>)
    %207 = fabric.instance @pe_invariant_i1_i0_to_o0(%64#8, %3#6) {sym_name = "pe_invariant_i1_i0_to_o0_r2_c3"} : (!dataflow.bits<1>, none) -> none
    %208 = fabric.instance @pe_invariant_i1_i32(%65#8, %107#8) {sym_name = "pe_invariant_i1_i32_r3_c1"} : (!dataflow.bits<1>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %209:2 = fabric.instance @pe_stream_i57_i57_i57_to_o57_o1(%146#6, %152#6, %147#10) {sym_name = "pe_stream_i57_i57_i57_to_o57_o1_r2_c0"} : (!dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>) -> (!dataflow.bits<57>, !dataflow.bits<1>)
    %210:2 = fabric.instance @pe_stream_i57_i57_i57_to_o57_o1(%147#8, %153#8, %148#9) {sym_name = "pe_stream_i57_i57_i57_to_o57_o1_r2_c1"} : (!dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>) -> (!dataflow.bits<57>, !dataflow.bits<1>)
    %211:2 = fabric.instance @pe_cond_br_i1_i0_to_o0_o0(%69#8, %4#6) {sym_name = "pe_cond_br_i1_i0_to_o0_o0_r3_c2"} : (!dataflow.bits<1>, none) -> (none, none)
    %212:2 = fabric.instance @pe_cond_br_i1_i0_to_o0_o0(%70#8, %5#6) {sym_name = "pe_cond_br_i1_i0_to_o0_o0_r3_c3"} : (!dataflow.bits<1>, none) -> (none, none)
    %213:2 = fabric.instance @pe_cond_br_i1_i0_to_o0_o0(%71#8, %7#6) {sym_name = "pe_cond_br_i1_i0_to_o0_o0_r3_c4"} : (!dataflow.bits<1>, none) -> (none, none)
    %214:2 = fabric.instance @pe_cond_br_i1_i0_to_o0_o0(%73#6, %8#8) {sym_name = "pe_cond_br_i1_i0_to_o0_o0_r4_c0"} : (!dataflow.bits<1>, none) -> (none, none)
    %215:2 = fabric.instance @pe_cond_br_i1_i32(%74#8, %108#8) {sym_name = "pe_cond_br_i1_i32_r3_c2"} : (!dataflow.bits<1>, !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>)
    %216:2 = fabric.instance @pe_cond_br_i1_i32(%75#8, %109#8) {sym_name = "pe_cond_br_i1_i32_r3_c3"} : (!dataflow.bits<1>, !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>)
    %217 = fabric.instance @pe_constant_i0_to_o32(%9#8) {sym_name = "pe_constant_i0_to_o32_r3_c4"} : (none) -> !dataflow.bits<32>
    %218 = fabric.instance @pe_constant_i0_to_o32(%10#8) {sym_name = "pe_constant_i0_to_o32_r3_c5"} : (none) -> !dataflow.bits<32>
    %219 = fabric.instance @pe_constant_i0_to_o32(%11#8) {sym_name = "pe_constant_i0_to_o32_r4_c0"} : (none) -> !dataflow.bits<32>
    %220 = fabric.instance @pe_constant_i0_to_o32(%12#8) {sym_name = "pe_constant_i0_to_o32_r4_c1"} : (none) -> !dataflow.bits<32>
    %221 = fabric.instance @pe_constant_i0_to_o32(%14#6) {sym_name = "pe_constant_i0_to_o32_r4_c2"} : (none) -> !dataflow.bits<32>
    %222 = fabric.instance @pe_constant_i0_to_o32(%15#8) {sym_name = "pe_constant_i0_to_o32_r4_c3"} : (none) -> !dataflow.bits<32>
    %223 = fabric.instance @pe_constant_i0_to_o57(%16#8) {sym_name = "pe_constant_i0_to_o57_r2_c2"} : (none) -> !dataflow.bits<57>
    %224 = fabric.instance @pe_constant_i0_to_o57(%17#8) {sym_name = "pe_constant_i0_to_o57_r2_c3"} : (none) -> !dataflow.bits<57>
    %225 = fabric.instance @pe_constant_i0_to_o57(%18#8) {sym_name = "pe_constant_i0_to_o57_r2_c4"} : (none) -> !dataflow.bits<57>
    %226 = fabric.instance @pe_constant_i0_to_o57(%19#8) {sym_name = "pe_constant_i0_to_o57_r3_c0"} : (none) -> !dataflow.bits<57>
    %227 = fabric.instance @pe_constant_i0_to_o57(%21#6) {sym_name = "pe_constant_i0_to_o57_r3_c1"} : (none) -> !dataflow.bits<57>
    %228 = fabric.instance @pe_constant_i0_to_o57(%22#8) {sym_name = "pe_constant_i0_to_o57_r3_c2"} : (none) -> !dataflow.bits<57>
    %229 = fabric.instance @pe_constant_i0_to_o64(%23#8) {sym_name = "pe_constant_i0_to_o64_r1_c1"} : (none) -> !dataflow.bits<64>
    %230 = fabric.instance @pe_join_i0(%24#8) {sym_name = "pe_join_i0_r3_c3"} : (none) -> none
    %231 = fabric.instance @pe_join_i0_i0(%25#8, %32#9) {sym_name = "pe_join_i0_i0_r3_c4"} : (none, none) -> none
    %232 = fabric.instance @pe_join_i0_i0_i0(%26#8, %33#8, %27#6) {sym_name = "pe_join_i0_i0_i0_r3_c5"} : (none, none, none) -> none
    %233 = fabric.instance @pe_mux_i57_i0_i0_to_o0(%155#8, %28#6, %35#6) {sym_name = "pe_mux_i57_i0_i0_to_o0_r3_c3"} : (!dataflow.bits<57>, none, none) -> none
    %234 = fabric.instance @pe_mux_i57_i0_i0_to_o0(%156#8, %29#8, %36#8) {sym_name = "pe_mux_i57_i0_i0_to_o0_r3_c4"} : (!dataflow.bits<57>, none, none) -> none
    %235 = fabric.instance @pe_mux_i57_i32_i32_to_o32(%158#6, %117#8, %124#8) {sym_name = "pe_mux_i57_i32_i32_to_o32_r4_c0"} : (!dataflow.bits<57>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    fabric.instance @pe_sink_i1(%76#8) {sym_name = "pe_sink_i1_r4_c3"} : (!dataflow.bits<1>) -> ()
    fabric.instance @pe_sink_i1(%77#8) {sym_name = "pe_sink_i1_r4_c4"} : (!dataflow.bits<1>) -> ()
    %236:2 = fabric.instance @load_pe_w32(%159#8, %118#8, %30#8) {sym_name = "load_pe_r4_c5"} : (!dataflow.bits<57>, !dataflow.bits<32>, none) -> (!dataflow.bits<32>, !dataflow.bits<57>)
    %237:2 = fabric.instance @load_pe_w32(%160#8, %120#6, %31#8) {sym_name = "load_pe_r5_c0"} : (!dataflow.bits<57>, !dataflow.bits<32>, none) -> (!dataflow.bits<32>, !dataflow.bits<57>)
    %238:2 = fabric.instance @store_pe_w32(%161#8, %121#8, %32#8) {sym_name = "store_pe_r5_c1"} : (!dataflow.bits<57>, !dataflow.bits<32>, none) -> (!dataflow.bits<32>, !dataflow.bits<57>)
    %239 = fabric.extmemory [ldCount = 0, stCount = 1, lsqDepth = 1] (%arg0, %133#4, %169#4) : memref<?xi32, strided<[1], offset: ?>>, (memref<?xi32, strided<[1], offset: ?>>, !dataflow.bits<32>, !dataflow.bits<57>) -> (none)
    %240:2 = fabric.extmemory [ldCount = 1, stCount = 0] (%arg1, %169#5) : memref<?xi32, strided<[1], offset: ?>>, (memref<?xi32, strided<[1], offset: ?>>, !dataflow.bits<57>) -> (!dataflow.bits<32>, none)
    %241:2 = fabric.memory [ldCount = 1, stCount = 0, is_private = true] (%169#6) : memref<256xi32>, (!dataflow.bits<57>) -> (!dataflow.bits<32>, none)
    fabric.yield %48#4 : none
  }
}
