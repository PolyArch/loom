module {
  fabric.pe @pe_addi_i32_i32(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.addi %arg0, %arg1 : i32
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
  fabric.module @genadg_2(%arg0: memref<?xi32, strided<[1], offset: ?>>, %arg1: memref<?xi32, strided<[1], offset: ?>>, %arg2: memref<?xi32, strided<[1], offset: ?>>, %arg3: none, %arg4: !dataflow.bits<32>, %arg5: !dataflow.bits<32>) -> (none, !dataflow.bits<32>) {
    %0:5 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111"]] %1#4, %1#5, %7#0, %7#1, %arg3, %201, %202#1, %203#1 : none -> none, none, none, none, none
    %1:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %2#4, %2#5, %8#0, %8#1, %0#0, %0#1 : none -> none, none, none, none, none, none, none
    %2:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %3#4, %3#5, %9#0, %9#1, %1#0, %1#1 : none -> none, none, none, none, none, none, none
    %3:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %4#4, %4#5, %10#0, %10#1, %2#0, %2#1 : none -> none, none, none, none, none, none, none
    %4:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %5#4, %5#5, %11#0, %11#1, %3#0, %3#1 : none -> none, none, none, none, none, none, none
    %5:7 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %6#2, %6#3, %12#0, %12#1, %4#0, %4#1, %173#1 : none -> none, none, none, none, none, none, none
    %6:4 = fabric.switch [connectivity_table = ["11111", "11111", "11111", "11111"]] %13#0, %13#1, %5#0, %5#1, %174#1 : none -> none, none, none, none
    %7:8 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %0#2, %0#3, %8#6, %8#7, %14#0, %14#1 : none -> none, none, none, none, none, none, none, none
    %8:10 = fabric.switch [connectivity_table = ["1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111"]] %1#2, %1#3, %9#6, %9#7, %15#0, %15#1, %7#2, %7#3, %165, %175#1 : none -> none, none, none, none, none, none, none, none, none, none
    %9:10 = fabric.switch [connectivity_table = ["1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111"]] %2#2, %2#3, %10#6, %10#7, %16#0, %16#1, %8#2, %8#3, %166, %176#1 : none -> none, none, none, none, none, none, none, none, none, none
    %10:9 = fabric.switch [connectivity_table = ["1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111"]] %3#2, %3#3, %11#6, %11#7, %17#0, %17#1, %9#2, %9#3, %167, %177#1 : none -> none, none, none, none, none, none, none, none, none
    %11:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %4#2, %4#3, %12#6, %12#7, %18#0, %18#1, %10#2, %10#3, %171 : none -> none, none, none, none, none, none, none, none, none
    %12:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %5#2, %5#3, %13#4, %13#5, %19#0, %19#1, %11#2, %11#3, %173#0 : none -> none, none, none, none, none, none, none, none, none
    %13:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %6#0, %6#1, %20#0, %20#1, %12#2, %12#3, %174#0 : none -> none, none, none, none, none, none
    %14:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %7#4, %7#5, %15#6, %15#7, %21#0, %21#1 : none -> none, none, none, none, none, none, none
    %15:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %8#4, %8#5, %16#6, %16#7, %22#0, %22#1, %14#2, %14#3, %175#0 : none -> none, none, none, none, none, none, none, none, none
    %16:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %9#4, %9#5, %17#6, %17#7, %23#0, %23#1, %15#2, %15#3, %176#0 : none -> none, none, none, none, none, none, none, none, none
    %17:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %10#4, %10#5, %18#6, %18#7, %24#0, %24#1, %16#2, %16#3, %177#0 : none -> none, none, none, none, none, none, none, none, none
    %18:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %11#4, %11#5, %19#6, %19#7, %25#0, %25#1, %17#2, %17#3 : none -> none, none, none, none, none, none, none, none, none
    %19:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %12#4, %12#5, %20#4, %20#5, %26#0, %26#1, %18#2, %18#3 : none -> none, none, none, none, none, none, none, none, none
    %20:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %13#2, %13#3, %27#0, %27#1, %19#2, %19#3 : none -> none, none, none, none, none, none
    %21:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %14#4, %14#5, %22#6, %22#7, %28#0, %28#1 : none -> none, none, none, none, none, none, none
    %22:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %15#4, %15#5, %23#6, %23#7, %29#0, %29#1, %21#2, %21#3 : none -> none, none, none, none, none, none, none, none, none
    %23:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %16#4, %16#5, %24#6, %24#7, %30#0, %30#1, %22#2, %22#3 : none -> none, none, none, none, none, none, none, none, none
    %24:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %17#4, %17#5, %25#6, %25#7, %31#0, %31#1, %23#2, %23#3 : none -> none, none, none, none, none, none, none, none, none
    %25:10 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %18#4, %18#5, %26#6, %26#7, %32#0, %32#1, %24#2, %24#3 : none -> none, none, none, none, none, none, none, none, none, none
    %26:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %19#4, %19#5, %27#4, %27#5, %33#0, %33#1, %25#2, %25#3 : none -> none, none, none, none, none, none, none, none, none
    %27:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %20#2, %20#3, %34#0, %34#1, %26#2, %26#3 : none -> none, none, none, none, none, none
    %28:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %21#4, %21#5, %29#6, %29#7, %35#0, %35#1 : none -> none, none, none, none, none, none, none
    %29:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %22#4, %22#5, %30#6, %30#7, %36#0, %36#1, %28#2, %28#3 : none -> none, none, none, none, none, none, none, none, none
    %30:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %23#4, %23#5, %31#6, %31#7, %37#0, %37#1, %29#2, %29#3, %189 : none -> none, none, none, none, none, none, none, none, none, none
    %31:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %24#4, %24#5, %32#6, %32#7, %38#0, %38#1, %30#2, %30#3, %190 : none -> none, none, none, none, none, none, none, none, none, none
    %32:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %25#4, %25#5, %33#6, %33#7, %39#0, %39#1, %31#2, %31#3, %191 : none -> none, none, none, none, none, none, none, none, none, none
    %33:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %26#4, %26#5, %34#4, %34#5, %40#0, %40#1, %32#2, %32#3, %192 : none -> none, none, none, none, none, none, none, none, none
    %34:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %27#2, %27#3, %41#0, %41#1, %33#2, %33#3, %193 : none -> none, none, none, none, none, none
    %35:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %28#4, %28#5, %36#6, %36#7, %42#0, %42#1 : none -> none, none, none, none, none, none, none
    %36:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %29#4, %29#5, %37#6, %37#7, %43#0, %43#1, %35#2, %35#3, %194 : none -> none, none, none, none, none, none, none, none, none
    %37:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %30#4, %30#5, %38#6, %38#7, %44#0, %44#1, %36#2, %36#3, %195 : none -> none, none, none, none, none, none, none, none
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
    %56:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %50#2, %50#3, %57#6, %57#7, %62#0, %62#1, %55#2, %55#3, %156 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %57:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %51#2, %51#3, %58#6, %58#7, %63#0, %63#1, %56#2, %56#3, %157 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %58:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %52#2, %52#3, %59#6, %59#7, %64#0, %64#1, %57#2, %57#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %59:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %53#2, %53#3, %60#4, %60#5, %65#0, %65#1, %58#2, %58#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %60:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %54#0, %54#1, %66#0, %66#1, %59#2, %59#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %61:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %55#4, %55#5, %62#6, %62#7, %67#0, %67#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %62:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %56#4, %56#5, %63#6, %63#7, %68#0, %68#1, %61#2, %61#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %63:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %57#4, %57#5, %64#6, %64#7, %69#0, %69#1, %62#2, %62#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %64:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %58#4, %58#5, %65#6, %65#7, %70#0, %70#1, %63#2, %63#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %65:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %59#4, %59#5, %66#4, %66#5, %71#0, %71#1, %64#2, %64#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %66:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %60#2, %60#3, %72#0, %72#1, %65#2, %65#3, %169#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %67:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %61#4, %61#5, %68#6, %68#7, %73#0, %73#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %68:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %62#4, %62#5, %69#6, %69#7, %74#0, %74#1, %67#2, %67#3, %170#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %69:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %63#4, %63#5, %70#6, %70#7, %75#0, %75#1, %68#2, %68#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %70:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %64#4, %64#5, %71#6, %71#7, %76#0, %76#1, %69#2, %69#3, %172#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %71:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %65#4, %65#5, %72#4, %72#5, %77#0, %77#1, %70#2, %70#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %72:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %66#2, %66#3, %78#0, %78#1, %71#2, %71#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %73:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %67#4, %67#5, %74#6, %74#7, %79#0, %79#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %74:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %68#4, %68#5, %75#6, %75#7, %80#0, %80#1, %73#2, %73#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %75:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %69#4, %69#5, %76#6, %76#7, %81#0, %81#1, %74#2, %74#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %76:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %70#4, %70#5, %77#6, %77#7, %82#0, %82#1, %75#2, %75#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %77:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %71#4, %71#5, %78#4, %78#5, %83#0, %83#1, %76#2, %76#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %78:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %72#2, %72#3, %84#0, %84#1, %77#2, %77#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %79:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %73#4, %73#5, %80#4, %80#5 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %80:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %74#4, %74#5, %81#4, %81#5, %79#2, %79#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %81:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %75#4, %75#5, %82#4, %82#5, %80#2, %80#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %82:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %76#4, %76#5, %83#4, %83#5, %81#2, %81#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %83:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %77#4, %77#5, %84#2, %84#3, %82#2, %82#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %84:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %78#2, %78#3, %83#2, %83#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %85:5 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111"]] %86#4, %86#5, %90#0, %90#1, %arg4, %arg5, %202#0, %203#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %86:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %87#4, %87#5, %91#0, %91#1, %85#0, %85#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %87:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %88#4, %88#5, %92#0, %92#1, %86#0, %86#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %88:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %89#2, %89#3, %93#0, %93#1, %87#0, %87#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %89:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %94#0, %94#1, %88#0, %88#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %90:8 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %85#2, %85#3, %91#6, %91#7, %95#0, %95#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %91:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %86#2, %86#3, %92#6, %92#7, %96#0, %96#1, %90#2, %90#3, %155 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %92:10 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %87#2, %87#3, %93#6, %93#7, %97#0, %97#1, %91#2, %91#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %93:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %88#2, %88#3, %94#4, %94#5, %98#0, %98#1, %92#2, %92#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %94:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %89#0, %89#1, %99#0, %99#1, %93#2, %93#3, %178#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %95:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %90#4, %90#5, %96#6, %96#7, %100#0, %100#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %96:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %91#4, %91#5, %97#6, %97#7, %101#0, %101#1, %95#2, %95#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %97:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %92#4, %92#5, %98#6, %98#7, %102#0, %102#1, %96#2, %96#3, %168 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %98:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %93#4, %93#5, %99#4, %99#5, %103#0, %103#1, %97#2, %97#3, %169#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %99:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %94#2, %94#3, %104#0, %104#1, %98#2, %98#3, %178#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %100:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %95#4, %95#5, %101#6, %101#7, %105#0, %105#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %101:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %96#4, %96#5, %102#6, %102#7, %106#0, %106#1, %100#2, %100#3, %179 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %102:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %97#4, %97#5, %103#6, %103#7, %107#0, %107#1, %101#2, %101#3, %180 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %103:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %98#4, %98#5, %104#4, %104#5, %108#0, %108#1, %102#2, %102#3, %196 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %104:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %99#2, %99#3, %109#0, %109#1, %103#2, %103#3, %197 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %105:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %100#4, %100#5, %106#4, %106#5 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %106:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %101#4, %101#5, %107#4, %107#5, %105#2, %105#3, %198#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %107:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %102#4, %102#5, %108#4, %108#5, %106#2, %106#3, %199#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %108:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %103#4, %103#5, %109#2, %109#3, %107#2, %107#3, %200#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %109:6 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111", "1111"]] %104#2, %104#3, %108#2, %108#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %110:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %111#4, %111#5, %116#0, %116#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %111:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %112#4, %112#5, %117#0, %117#1, %110#0, %110#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %112:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %113#4, %113#5, %118#0, %118#1, %111#0, %111#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %113:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %114#4, %114#5, %119#0, %119#1, %112#0, %112#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %114:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %115#2, %115#3, %120#0, %120#1, %113#0, %113#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %115:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %121#0, %121#1, %114#0, %114#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %116:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %110#2, %110#3, %117#6, %117#7, %122#0, %122#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %117:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %111#2, %111#3, %118#6, %118#7, %123#0, %123#1, %116#2, %116#3, %159 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %118:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %112#2, %112#3, %119#6, %119#7, %124#0, %124#1, %117#2, %117#3, %160 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %119:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %113#2, %113#3, %120#6, %120#7, %125#0, %125#1, %118#2, %118#3, %161 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %120:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %114#2, %114#3, %121#4, %121#5, %126#0, %126#1, %119#2, %119#3, %162 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %121:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %115#0, %115#1, %127#0, %127#1, %120#2, %120#3, %163 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %122:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %116#4, %116#5, %123#6, %123#7, %128#0, %128#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %123:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %117#4, %117#5, %124#6, %124#7, %129#0, %129#1, %122#2, %122#3, %164 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %124:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %118#4, %118#5, %125#6, %125#7, %130#0, %130#1, %123#2, %123#3, %170#0 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %125:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %119#4, %119#5, %126#6, %126#7, %131#0, %131#1, %124#2, %124#3, %172#0 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %126:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %120#4, %120#5, %127#4, %127#5, %132#0, %132#1, %125#2, %125#3, %181 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %127:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %121#2, %121#3, %133#0, %133#1, %126#2, %126#3, %182 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %128:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %122#4, %122#5, %129#6, %129#7, %134#0, %134#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %129:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %123#4, %123#5, %130#6, %130#7, %135#0, %135#1, %128#2, %128#3, %183 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %130:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %124#4, %124#5, %131#6, %131#7, %136#0, %136#1, %129#2, %129#3, %184 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %131:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %125#4, %125#5, %132#6, %132#7, %137#0, %137#1, %130#2, %130#3, %185 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %132:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %126#4, %126#5, %133#4, %133#5, %138#0, %138#1, %131#2, %131#3, %186 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %133:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %127#2, %127#3, %139#0, %139#1, %132#2, %132#3, %187 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %134:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %128#4, %128#5, %135#6, %135#7, %140#0, %140#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %135:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %129#4, %129#5, %136#6, %136#7, %141#0, %141#1, %134#2, %134#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %136:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %130#4, %130#5, %137#6, %137#7, %142#0, %142#1, %135#2, %135#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %137:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %131#4, %131#5, %138#6, %138#7, %143#0, %143#1, %136#2, %136#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %138:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %132#4, %132#5, %139#4, %139#5, %144#0, %144#1, %137#2, %137#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %139:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %133#2, %133#3, %145#0, %145#1, %138#2, %138#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %140:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %134#4, %134#5, %141#4, %141#5 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %141:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %135#4, %135#5, %142#4, %142#5, %140#2, %140#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %142:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %136#4, %136#5, %143#4, %143#5, %141#2, %141#3, %198#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %143:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %137#4, %137#5, %144#4, %144#5, %142#2, %142#3, %199#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %144:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %138#4, %138#5, %145#2, %145#3, %143#2, %143#3, %200#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %145:7 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111", "1111", "1111"]] %139#2, %139#3, %144#2, %144#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %146:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %147#4, %147#5, %149#0, %149#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %147:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %148#2, %148#3, %150#0, %150#1, %146#0, %146#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %148:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %151#0, %151#1, %147#0, %147#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %149:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %146#2, %146#3, %150#6, %150#7, %152#0, %152#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %150:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %147#2, %147#3, %151#4, %151#5, %153#0, %153#1, %149#2, %149#3, %158 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %151:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %148#0, %148#1, %154#0, %154#1, %150#2, %150#3 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %152:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %149#4, %149#5, %153#4, %153#5 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %153:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %150#4, %150#5, %154#2, %154#3, %152#2, %152#3 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %154:4 = fabric.switch [connectivity_table = ["11111", "11111", "11111", "11111"]] %151#2, %151#3, %153#2, %153#3, %188 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %155 = fabric.instance @pe_addi_i32_i32(%85#4, %90#7) {sym_name = "pe_addi_i32_i32_r0_c0"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %156 = fabric.instance @pe_cmpi_i32_i32_to_o1(%86#6, %91#9) {sym_name = "pe_cmpi_i32_i32_to_o1_r0_c1"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<1>
    %157 = fabric.instance @pe_cmpi_i32_i32_to_o1(%87#6, %92#9) {sym_name = "pe_cmpi_i32_i32_to_o1_r0_c2"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<1>
    %158 = fabric.instance @pe_extui_i32_to_o64(%88#6) {sym_name = "pe_extui_i32_to_o64_r0_c3"} : (!dataflow.bits<32>) -> !dataflow.bits<64>
    %159 = fabric.instance @pe_index_cast_i32_to_o57(%90#6) {sym_name = "pe_index_cast_i32_to_o57_r1_c0"} : (!dataflow.bits<32>) -> !dataflow.bits<57>
    %160 = fabric.instance @pe_index_cast_i64_to_o57(%147#6) {sym_name = "pe_index_cast_i64_to_o57_r0_c1"} : (!dataflow.bits<64>) -> !dataflow.bits<57>
    %161 = fabric.instance @pe_index_cast_i64_to_o57(%149#6) {sym_name = "pe_index_cast_i64_to_o57_r1_c0"} : (!dataflow.bits<64>) -> !dataflow.bits<57>
    %162 = fabric.instance @pe_select_i1_i57_i57(%51#6, %113#6, %119#8) {sym_name = "pe_select_i1_i57_i57_r0_c3"} : (!dataflow.bits<1>, !dataflow.bits<57>, !dataflow.bits<57>) -> !dataflow.bits<57>
    %163 = fabric.instance @pe_select_i1_i57_i57(%52#6, %114#6, %120#8) {sym_name = "pe_select_i1_i57_i57_r0_c4"} : (!dataflow.bits<1>, !dataflow.bits<57>, !dataflow.bits<57>) -> !dataflow.bits<57>
    %164 = fabric.instance @pe_select_i1_i57_i57(%53#6, %116#6, %122#6) {sym_name = "pe_select_i1_i57_i57_r1_c0"} : (!dataflow.bits<1>, !dataflow.bits<57>, !dataflow.bits<57>) -> !dataflow.bits<57>
    %165 = fabric.instance @pe_carry_i1_i0_i0_to_o0(%55#6, %0#4, %7#7) {sym_name = "pe_carry_i1_i0_i0_to_o0_r1_c0"} : (!dataflow.bits<1>, none, none) -> none
    %166 = fabric.instance @pe_carry_i1_i0_i0_to_o0(%56#8, %1#6, %8#9) {sym_name = "pe_carry_i1_i0_i0_to_o0_r1_c1"} : (!dataflow.bits<1>, none, none) -> none
    %167 = fabric.instance @pe_carry_i1_i0_i0_to_o0(%57#8, %2#6, %9#9) {sym_name = "pe_carry_i1_i0_i0_to_o0_r1_c2"} : (!dataflow.bits<1>, none, none) -> none
    %168 = fabric.instance @pe_carry_i1_i32_i32(%58#8, %91#8, %96#8) {sym_name = "pe_carry_i1_i32_i32_r1_c1"} : (!dataflow.bits<1>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %169:2 = fabric.instance @pe_gate_i32_i1_to_o32_o1(%92#8, %59#8) {sym_name = "pe_gate_i32_i1_to_o32_o1_r1_c2"} : (!dataflow.bits<32>, !dataflow.bits<1>) -> (!dataflow.bits<32>, !dataflow.bits<1>)
    %170:2 = fabric.instance @pe_gate_i57_i1_to_o57_o1(%117#8, %61#6) {sym_name = "pe_gate_i57_i1_to_o57_o1_r1_c1"} : (!dataflow.bits<57>, !dataflow.bits<1>) -> (!dataflow.bits<57>, !dataflow.bits<1>)
    %171 = fabric.instance @pe_invariant_i1_i0_to_o0(%62#8, %3#6) {sym_name = "pe_invariant_i1_i0_to_o0_r2_c1"} : (!dataflow.bits<1>, none) -> none
    %172:2 = fabric.instance @pe_stream_i57_i57_i57_to_o57_o1(%118#8, %124#8, %119#9) {sym_name = "pe_stream_i57_i57_i57_to_o57_o1_r1_c2"} : (!dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>) -> (!dataflow.bits<57>, !dataflow.bits<1>)
    %173:2 = fabric.instance @pe_cond_br_i1_i0_to_o0_o0(%64#8, %4#6) {sym_name = "pe_cond_br_i1_i0_to_o0_o0_r2_c3"} : (!dataflow.bits<1>, none) -> (none, none)
    %174:2 = fabric.instance @pe_cond_br_i1_i0_to_o0_o0(%65#8, %5#6) {sym_name = "pe_cond_br_i1_i0_to_o0_o0_r2_c4"} : (!dataflow.bits<1>, none) -> (none, none)
    %175:2 = fabric.instance @pe_cond_br_i1_i0_to_o0_o0(%67#6, %7#6) {sym_name = "pe_cond_br_i1_i0_to_o0_o0_r3_c0"} : (!dataflow.bits<1>, none) -> (none, none)
    %176:2 = fabric.instance @pe_cond_br_i1_i0_to_o0_o0(%68#8, %8#8) {sym_name = "pe_cond_br_i1_i0_to_o0_o0_r3_c1"} : (!dataflow.bits<1>, none) -> (none, none)
    %177:2 = fabric.instance @pe_cond_br_i1_i0_to_o0_o0(%69#8, %9#8) {sym_name = "pe_cond_br_i1_i0_to_o0_o0_r3_c2"} : (!dataflow.bits<1>, none) -> (none, none)
    %178:2 = fabric.instance @pe_cond_br_i1_i32(%70#8, %93#8) {sym_name = "pe_cond_br_i1_i32_r1_c3"} : (!dataflow.bits<1>, !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>)
    %179 = fabric.instance @pe_constant_i0_to_o32(%10#8) {sym_name = "pe_constant_i0_to_o32_r2_c0"} : (none) -> !dataflow.bits<32>
    %180 = fabric.instance @pe_constant_i0_to_o32(%11#8) {sym_name = "pe_constant_i0_to_o32_r2_c1"} : (none) -> !dataflow.bits<32>
    %181 = fabric.instance @pe_constant_i0_to_o57(%12#8) {sym_name = "pe_constant_i0_to_o57_r1_c3"} : (none) -> !dataflow.bits<57>
    %182 = fabric.instance @pe_constant_i0_to_o57(%14#6) {sym_name = "pe_constant_i0_to_o57_r1_c4"} : (none) -> !dataflow.bits<57>
    %183 = fabric.instance @pe_constant_i0_to_o57(%15#8) {sym_name = "pe_constant_i0_to_o57_r2_c0"} : (none) -> !dataflow.bits<57>
    %184 = fabric.instance @pe_constant_i0_to_o57(%16#8) {sym_name = "pe_constant_i0_to_o57_r2_c1"} : (none) -> !dataflow.bits<57>
    %185 = fabric.instance @pe_constant_i0_to_o57(%17#8) {sym_name = "pe_constant_i0_to_o57_r2_c2"} : (none) -> !dataflow.bits<57>
    %186 = fabric.instance @pe_constant_i0_to_o57(%18#8) {sym_name = "pe_constant_i0_to_o57_r2_c3"} : (none) -> !dataflow.bits<57>
    %187 = fabric.instance @pe_constant_i0_to_o57(%19#8) {sym_name = "pe_constant_i0_to_o57_r2_c4"} : (none) -> !dataflow.bits<57>
    %188 = fabric.instance @pe_constant_i0_to_o64(%21#6) {sym_name = "pe_constant_i0_to_o64_r1_c1"} : (none) -> !dataflow.bits<64>
    %189 = fabric.instance @pe_join_i0(%22#8) {sym_name = "pe_join_i0_r3_c1"} : (none) -> none
    %190 = fabric.instance @pe_join_i0_i0(%23#8, %30#9) {sym_name = "pe_join_i0_i0_r3_c2"} : (none, none) -> none
    %191 = fabric.instance @pe_join_i0_i0_i0(%24#8, %31#9, %25#9) {sym_name = "pe_join_i0_i0_i0_r3_c3"} : (none, none, none) -> none
    %192 = fabric.instance @pe_mux_i57_i0_i0_to_o0(%128#6, %25#8, %32#9) {sym_name = "pe_mux_i57_i0_i0_to_o0_r3_c0"} : (!dataflow.bits<57>, none, none) -> none
    %193 = fabric.instance @pe_mux_i57_i0_i0_to_o0(%129#8, %26#8, %33#8) {sym_name = "pe_mux_i57_i0_i0_to_o0_r3_c1"} : (!dataflow.bits<57>, none, none) -> none
    %194 = fabric.instance @pe_mux_i57_i0_i0_to_o0(%130#8, %28#6, %35#6) {sym_name = "pe_mux_i57_i0_i0_to_o0_r3_c2"} : (!dataflow.bits<57>, none, none) -> none
    %195 = fabric.instance @pe_mux_i57_i0_i0_to_o0(%131#8, %29#8, %36#8) {sym_name = "pe_mux_i57_i0_i0_to_o0_r3_c3"} : (!dataflow.bits<57>, none, none) -> none
    %196 = fabric.instance @pe_mux_i57_i32_i32_to_o32(%132#8, %97#8, %102#9) {sym_name = "pe_mux_i57_i32_i32_to_o32_r3_c4"} : (!dataflow.bits<57>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %197 = fabric.instance @pe_mux_i57_i32_i32_to_o32(%134#6, %98#8, %103#8) {sym_name = "pe_mux_i57_i32_i32_to_o32_r4_c0"} : (!dataflow.bits<57>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    fabric.instance @pe_sink_i1(%71#8) {sym_name = "pe_sink_i1_r3_c4"} : (!dataflow.bits<1>) -> ()
    %198:2 = fabric.instance @load_pe_w32(%135#8, %100#6, %30#8) {sym_name = "load_pe_r3_c0"} : (!dataflow.bits<57>, !dataflow.bits<32>, none) -> (!dataflow.bits<32>, !dataflow.bits<57>)
    %199:2 = fabric.instance @load_pe_w32(%136#8, %101#8, %31#8) {sym_name = "load_pe_r3_c1"} : (!dataflow.bits<57>, !dataflow.bits<32>, none) -> (!dataflow.bits<32>, !dataflow.bits<57>)
    %200:2 = fabric.instance @store_pe_w32(%137#8, %102#8, %32#8) {sym_name = "store_pe_r3_c2"} : (!dataflow.bits<57>, !dataflow.bits<32>, none) -> (!dataflow.bits<32>, !dataflow.bits<57>)
    %201 = fabric.extmemory [ldCount = 0, stCount = 1, lsqDepth = 1] (%arg0, %109#5, %145#4) : memref<?xi32, strided<[1], offset: ?>>, (memref<?xi32, strided<[1], offset: ?>>, !dataflow.bits<32>, !dataflow.bits<57>) -> (none)
    %202:2 = fabric.extmemory [ldCount = 1, stCount = 0] (%arg1, %145#5) : memref<?xi32, strided<[1], offset: ?>>, (memref<?xi32, strided<[1], offset: ?>>, !dataflow.bits<57>) -> (!dataflow.bits<32>, none)
    %203:2 = fabric.extmemory [ldCount = 1, stCount = 0] (%arg2, %145#6) : memref<?xi32, strided<[1], offset: ?>>, (memref<?xi32, strided<[1], offset: ?>>, !dataflow.bits<57>) -> (!dataflow.bits<32>, none)
    fabric.yield %48#4, %109#4 : none, !dataflow.bits<32>
  }
}
