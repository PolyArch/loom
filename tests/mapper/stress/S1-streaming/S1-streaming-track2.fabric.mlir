module {
  fabric.pe @pe_addf_i32_i32(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = arith.addf %arg0, %arg1 : f32
    fabric.yield %0 : f32
  }
  fabric.pe @pe_addi_i32_i32(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.addi %arg0, %arg1 : i32
    fabric.yield %0 : i32
  }
  fabric.pe @pe_cmpf_i32_i32_to_o1(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<1>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = arith.cmpf false, %arg0, %arg1 : f32
    fabric.yield %0 : i1
  }
  fabric.pe @pe_cmpi_i32_i32_to_o1(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<1>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.cmpi eq, %arg0, %arg1 : i32
    fabric.yield %0 : i1
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
  fabric.pe @pe_select_i1_i32_i32(%arg0: !dataflow.bits<1>, %arg1: !dataflow.bits<32>, %arg2: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i1, %arg1: i32, %arg2: i32):
    %0 = arith.select %arg0, %arg1, %arg2 : i32
    fabric.yield %0 : i32
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
  fabric.pe @load_pe_w32f(%arg0: !dataflow.bits<57>, %arg1: !dataflow.bits<32>, %arg2: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>, !dataflow.bits<57>) {
  ^bb0(%arg0: index, %arg1: f32, %arg2: none):
    %dataResult, %addressResults = handshake.load [%arg0] %arg1, %arg2 : index, f32
    fabric.yield %dataResult, %addressResults : f32, index
  }
  fabric.pe @store_pe_w32f(%arg0: !dataflow.bits<57>, %arg1: !dataflow.bits<32>, %arg2: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>, !dataflow.bits<57>) {
  ^bb0(%arg0: index, %arg1: f32, %arg2: none):
    %dataResult, %addressResult = handshake.store [%arg0] %arg1, %arg2 : index, f32
    fabric.yield %dataResult, %addressResult : f32, index
  }
  fabric.module @genadg_3(%arg0: memref<?xf32, strided<[1], offset: ?>>, %arg1: memref<?xi32, strided<[1], offset: ?>>, %arg2: memref<?xf32, strided<[1], offset: ?>>, %arg3: none, %arg4: !dataflow.bits<32>, %arg5: !dataflow.bits<32>) -> (none, !dataflow.bits<32>) {
    %0:5 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111"]] %1#4, %1#5, %6#0, %6#1, %arg3, %152, %153#1, %154#1 : none -> none, none, none, none, none
    %1:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %2#4, %2#5, %7#0, %7#1, %0#0, %0#1 : none -> none, none, none, none, none, none, none
    %2:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %3#4, %3#5, %8#0, %8#1, %1#0, %1#1 : none -> none, none, none, none, none, none, none
    %3:7 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %4#4, %4#5, %9#0, %9#1, %2#0, %2#1, %134#1 : none -> none, none, none, none, none, none, none
    %4:7 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %5#2, %5#3, %10#0, %10#1, %3#0, %3#1, %135#1 : none -> none, none, none, none, none, none, none
    %5:4 = fabric.switch [connectivity_table = ["11111", "11111", "11111", "11111"]] %11#0, %11#1, %4#0, %4#1, %136#1 : none -> none, none, none, none
    %6:8 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %0#2, %0#3, %7#6, %7#7, %12#0, %12#1 : none -> none, none, none, none, none, none, none, none
    %7:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %1#2, %1#3, %8#6, %8#7, %13#0, %13#1, %6#2, %6#3, %128 : none -> none, none, none, none, none, none, none, none, none, none
    %8:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %2#2, %2#3, %9#6, %9#7, %14#0, %14#1, %7#2, %7#3, %129 : none -> none, none, none, none, none, none, none, none, none
    %9:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %3#2, %3#3, %10#6, %10#7, %15#0, %15#1, %8#2, %8#3, %134#0 : none -> none, none, none, none, none, none, none, none, none
    %10:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %4#2, %4#3, %11#4, %11#5, %16#0, %16#1, %9#2, %9#3, %135#0 : none -> none, none, none, none, none, none, none, none, none
    %11:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %5#0, %5#1, %17#0, %17#1, %10#2, %10#3, %136#0 : none -> none, none, none, none, none, none
    %12:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %6#4, %6#5, %13#6, %13#7, %18#0, %18#1 : none -> none, none, none, none, none, none, none
    %13:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %7#4, %7#5, %14#6, %14#7, %19#0, %19#1, %12#2, %12#3 : none -> none, none, none, none, none, none, none, none, none
    %14:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %8#4, %8#5, %15#6, %15#7, %20#0, %20#1, %13#2, %13#3 : none -> none, none, none, none, none, none, none, none, none
    %15:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %9#4, %9#5, %16#6, %16#7, %21#0, %21#1, %14#2, %14#3 : none -> none, none, none, none, none, none, none, none, none
    %16:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %10#4, %10#5, %17#4, %17#5, %22#0, %22#1, %15#2, %15#3 : none -> none, none, none, none, none, none, none, none, none
    %17:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %11#2, %11#3, %23#0, %23#1, %16#2, %16#3 : none -> none, none, none, none, none, none
    %18:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %12#4, %12#5, %19#6, %19#7, %24#0, %24#1 : none -> none, none, none, none, none, none, none
    %19:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %13#4, %13#5, %20#6, %20#7, %25#0, %25#1, %18#2, %18#3 : none -> none, none, none, none, none, none, none, none, none
    %20:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %14#4, %14#5, %21#6, %21#7, %26#0, %26#1, %19#2, %19#3, %144 : none -> none, none, none, none, none, none, none, none, none, none
    %21:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %15#4, %15#5, %22#6, %22#7, %27#0, %27#1, %20#2, %20#3, %145 : none -> none, none, none, none, none, none, none, none, none
    %22:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %16#4, %16#5, %23#4, %23#5, %28#0, %28#1, %21#2, %21#3, %146 : none -> none, none, none, none, none, none, none, none, none
    %23:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %17#2, %17#3, %29#0, %29#1, %22#2, %22#3, %147 : none -> none, none, none, none, none, none
    %24:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %18#4, %18#5, %25#6, %25#7, %30#0, %30#1 : none -> none, none, none, none, none, none
    %25:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %19#4, %19#5, %26#6, %26#7, %31#0, %31#1, %24#2, %24#3 : none -> none, none, none, none, none, none, none, none
    %26:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %20#4, %20#5, %27#6, %27#7, %32#0, %32#1, %25#2, %25#3 : none -> none, none, none, none, none, none, none, none
    %27:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %21#4, %21#5, %28#6, %28#7, %33#0, %33#1, %26#2, %26#3 : none -> none, none, none, none, none, none, none, none
    %28:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %22#4, %22#5, %29#4, %29#5, %34#0, %34#1, %27#2, %27#3 : none -> none, none, none, none, none, none, none, none
    %29:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %23#2, %23#3, %35#0, %35#1, %28#2, %28#3 : none -> none, none, none, none, none, none
    %30:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %24#4, %24#5, %31#4, %31#5 : none -> none, none, none, none
    %31:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %25#4, %25#5, %32#4, %32#5, %30#2, %30#3 : none -> none, none, none, none, none, none
    %32:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %26#4, %26#5, %33#4, %33#5, %31#2, %31#3 : none -> none, none, none, none, none, none
    %33:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %27#4, %27#5, %34#4, %34#5, %32#2, %32#3 : none -> none, none, none, none, none, none
    %34:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %28#4, %28#5, %35#2, %35#3, %33#2, %33#3 : none -> none, none, none, none, none, none
    %35:5 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111"]] %29#2, %29#3, %34#2, %34#3 : none -> none, none, none, none, none
    %36:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %37#4, %37#5, %41#0, %41#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %37:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %38#4, %38#5, %42#0, %42#1, %36#0, %36#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %38:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %39#4, %39#5, %43#0, %43#1, %37#0, %37#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %39:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %40#2, %40#3, %44#0, %44#1, %38#0, %38#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %40:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %45#0, %45#1, %39#0, %39#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %41:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %36#2, %36#3, %42#6, %42#7, %46#0, %46#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %42:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %37#2, %37#3, %43#6, %43#7, %47#0, %47#1, %41#2, %41#3, %122 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %43:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %38#2, %38#3, %44#6, %44#7, %48#0, %48#1, %42#2, %42#3, %123 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %44:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %39#2, %39#3, %45#4, %45#5, %49#0, %49#1, %43#2, %43#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %45:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %40#0, %40#1, %50#0, %50#1, %44#2, %44#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %46:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %41#4, %41#5, %47#6, %47#7, %51#0, %51#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %47:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %42#4, %42#5, %48#6, %48#7, %52#0, %52#1, %46#2, %46#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %48:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %43#4, %43#5, %49#6, %49#7, %53#0, %53#1, %47#2, %47#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %49:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %44#4, %44#5, %50#4, %50#5, %54#0, %54#1, %48#2, %48#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %50:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %45#2, %45#3, %55#0, %55#1, %49#2, %49#3, %131#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %51:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %46#4, %46#5, %52#6, %52#7, %56#0, %56#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %52:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %47#4, %47#5, %53#6, %53#7, %57#0, %57#1, %51#2, %51#3, %132#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %53:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %48#4, %48#5, %54#6, %54#7, %58#0, %58#1, %52#2, %52#3, %133#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %54:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %49#4, %49#5, %55#4, %55#5, %59#0, %59#1, %53#2, %53#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %55:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %50#2, %50#3, %60#0, %60#1, %54#2, %54#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %56:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %51#4, %51#5, %57#4, %57#5 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %57:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %52#4, %52#5, %58#4, %58#5, %56#2, %56#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %58:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %53#4, %53#5, %59#4, %59#5, %57#2, %57#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %59:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %54#4, %54#5, %60#2, %60#3, %58#2, %58#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %60:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %55#2, %55#3, %59#2, %59#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %61:5 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111"]] %62#4, %62#5, %66#0, %66#1, %arg4, %arg5, %153#0, %154#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %62:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %63#4, %63#5, %67#0, %67#1, %61#0, %61#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %63:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %64#4, %64#5, %68#0, %68#1, %62#0, %62#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %64:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %65#2, %65#3, %69#0, %69#1, %63#0, %63#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %65:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %70#0, %70#1, %64#0, %64#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %66:8 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %61#2, %61#3, %67#6, %67#7, %71#0, %71#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %67:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %62#2, %62#3, %68#6, %68#7, %72#0, %72#1, %66#2, %66#3, %120 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %68:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %63#2, %63#3, %69#6, %69#7, %73#0, %73#1, %67#2, %67#3, %121 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %69:10 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %64#2, %64#3, %70#4, %70#5, %74#0, %74#1, %68#2, %68#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %70:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %65#0, %65#1, %75#0, %75#1, %69#2, %69#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %71:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %66#4, %66#5, %72#6, %72#7, %76#0, %76#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %72:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %67#4, %67#5, %73#6, %73#7, %77#0, %77#1, %71#2, %71#3, %137#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %73:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %68#4, %68#5, %74#6, %74#7, %78#0, %78#1, %72#2, %72#3, %126 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %74:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %69#4, %69#5, %75#4, %75#5, %79#0, %79#1, %73#2, %73#3, %130 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %75:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %70#2, %70#3, %80#0, %80#1, %74#2, %74#3, %131#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %76:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %71#4, %71#5, %77#6, %77#7, %81#0, %81#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %77:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %72#4, %72#5, %78#6, %78#7, %82#0, %82#1, %76#2, %76#3, %137#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %78:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %73#4, %73#5, %79#6, %79#7, %83#0, %83#1, %77#2, %77#3, %138 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %79:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %74#4, %74#5, %80#4, %80#5, %84#0, %84#1, %78#2, %78#3, %139 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %80:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %75#2, %75#3, %85#0, %85#1, %79#2, %79#3, %148 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %81:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %76#4, %76#5, %82#4, %82#5 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %82:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %77#4, %77#5, %83#4, %83#5, %81#2, %81#3, %149#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %83:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %78#4, %78#5, %84#4, %84#5, %82#2, %82#3, %150#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %84:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %79#4, %79#5, %85#2, %85#3, %83#2, %83#3, %151#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %85:6 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111", "1111"]] %80#2, %80#3, %84#2, %84#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %86:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %87#4, %87#5, %91#0, %91#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %87:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %88#4, %88#5, %92#0, %92#1, %86#0, %86#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %88:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %89#4, %89#5, %93#0, %93#1, %87#0, %87#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %89:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %90#2, %90#3, %94#0, %94#1, %88#0, %88#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %90:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %95#0, %95#1, %89#0, %89#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %91:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %86#2, %86#3, %92#6, %92#7, %96#0, %96#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %92:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %87#2, %87#3, %93#6, %93#7, %97#0, %97#1, %91#2, %91#3, %124 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %93:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %88#2, %88#3, %94#6, %94#7, %98#0, %98#1, %92#2, %92#3, %125 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %94:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %89#2, %89#3, %95#4, %95#5, %99#0, %99#1, %93#2, %93#3, %127 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %95:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %90#0, %90#1, %100#0, %100#1, %94#2, %94#3, %132#0 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %96:8 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %91#4, %91#5, %97#6, %97#7, %101#0, %101#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %97:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %92#4, %92#5, %98#6, %98#7, %102#0, %102#1, %96#2, %96#3, %133#0 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %98:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %93#4, %93#5, %99#6, %99#7, %103#0, %103#1, %97#2, %97#3, %140 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %99:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %94#4, %94#5, %100#4, %100#5, %104#0, %104#1, %98#2, %98#3, %141 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %100:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %95#2, %95#3, %105#0, %105#1, %99#2, %99#3, %142 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %101:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %96#4, %96#5, %102#6, %102#7, %106#0, %106#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %102:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %97#4, %97#5, %103#6, %103#7, %107#0, %107#1, %101#2, %101#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %103:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %98#4, %98#5, %104#6, %104#7, %108#0, %108#1, %102#2, %102#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %104:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %99#4, %99#5, %105#4, %105#5, %109#0, %109#1, %103#2, %103#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %105:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %100#2, %100#3, %110#0, %110#1, %104#2, %104#3, %149#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %106:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %101#4, %101#5, %107#4, %107#5 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %107:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %102#4, %102#5, %108#4, %108#5, %106#2, %106#3, %150#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %108:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %103#4, %103#5, %109#4, %109#5, %107#2, %107#3, %151#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %109:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %104#4, %104#5, %110#2, %110#3, %108#2, %108#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %110:7 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111", "1111", "1111"]] %105#2, %105#3, %109#2, %109#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %111:5 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111"]] %112#4, %112#5, %114#0, %114#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %112:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %113#2, %113#3, %115#0, %115#1, %111#0, %111#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %113:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %116#0, %116#1, %112#0, %112#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %114:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %111#2, %111#3, %115#6, %115#7, %117#0, %117#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %115:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %112#2, %112#3, %116#4, %116#5, %118#0, %118#1, %114#2, %114#3 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %116:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %113#0, %113#1, %119#0, %119#1, %115#2, %115#3, %143 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %117:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %114#4, %114#5, %118#4, %118#5 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %118:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %115#4, %115#5, %119#2, %119#3, %117#2, %117#3 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %119:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %116#2, %116#3, %118#2, %118#3 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %120 = fabric.instance @pe_addf_i32_i32(%61#4, %66#7) {sym_name = "pe_addf_i32_i32_r0_c0"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %121 = fabric.instance @pe_addi_i32_i32(%62#6, %67#9) {sym_name = "pe_addi_i32_i32_r0_c1"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %122 = fabric.instance @pe_cmpf_i32_i32_to_o1(%63#6, %68#9) {sym_name = "pe_cmpf_i32_i32_to_o1_r0_c2"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<1>
    %123 = fabric.instance @pe_cmpi_i32_i32_to_o1(%64#6, %69#9) {sym_name = "pe_cmpi_i32_i32_to_o1_r0_c3"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<1>
    %124 = fabric.instance @pe_index_cast_i32_to_o57(%66#6) {sym_name = "pe_index_cast_i32_to_o57_r1_c0"} : (!dataflow.bits<32>) -> !dataflow.bits<57>
    %125 = fabric.instance @pe_index_cast_i64_to_o57(%111#4) {sym_name = "pe_index_cast_i64_to_o57_r0_c0"} : (!dataflow.bits<64>) -> !dataflow.bits<57>
    %126 = fabric.instance @pe_select_i1_i32_i32(%38#6, %67#8, %72#8) {sym_name = "pe_select_i1_i32_i32_r1_c1"} : (!dataflow.bits<1>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %127 = fabric.instance @pe_select_i1_i57_i57(%39#6, %88#6, %93#8) {sym_name = "pe_select_i1_i57_i57_r0_c2"} : (!dataflow.bits<1>, !dataflow.bits<57>, !dataflow.bits<57>) -> !dataflow.bits<57>
    %128 = fabric.instance @pe_carry_i1_i0_i0_to_o0(%41#6, %0#4, %6#7) {sym_name = "pe_carry_i1_i0_i0_to_o0_r1_c0"} : (!dataflow.bits<1>, none, none) -> none
    %129 = fabric.instance @pe_carry_i1_i0_i0_to_o0(%42#8, %1#6, %7#9) {sym_name = "pe_carry_i1_i0_i0_to_o0_r1_c1"} : (!dataflow.bits<1>, none, none) -> none
    %130 = fabric.instance @pe_carry_i1_i32_i32(%43#8, %68#8, %73#8) {sym_name = "pe_carry_i1_i32_i32_r1_c2"} : (!dataflow.bits<1>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %131:2 = fabric.instance @pe_gate_i32_i1_to_o32_o1(%69#8, %44#8) {sym_name = "pe_gate_i32_i1_to_o32_o1_r1_c3"} : (!dataflow.bits<32>, !dataflow.bits<1>) -> (!dataflow.bits<32>, !dataflow.bits<1>)
    %132:2 = fabric.instance @pe_gate_i57_i1_to_o57_o1(%89#6, %46#6) {sym_name = "pe_gate_i57_i1_to_o57_o1_r0_c3"} : (!dataflow.bits<57>, !dataflow.bits<1>) -> (!dataflow.bits<57>, !dataflow.bits<1>)
    %133:2 = fabric.instance @pe_stream_i57_i57_i57_to_o57_o1(%91#6, %96#7, %92#8) {sym_name = "pe_stream_i57_i57_i57_to_o57_o1_r1_c0"} : (!dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>) -> (!dataflow.bits<57>, !dataflow.bits<1>)
    %134:2 = fabric.instance @pe_cond_br_i1_i0_to_o0_o0(%48#8, %2#6) {sym_name = "pe_cond_br_i1_i0_to_o0_o0_r2_c2"} : (!dataflow.bits<1>, none) -> (none, none)
    %135:2 = fabric.instance @pe_cond_br_i1_i0_to_o0_o0(%49#8, %3#6) {sym_name = "pe_cond_br_i1_i0_to_o0_o0_r2_c3"} : (!dataflow.bits<1>, none) -> (none, none)
    %136:2 = fabric.instance @pe_cond_br_i1_i0_to_o0_o0(%51#6, %4#6) {sym_name = "pe_cond_br_i1_i0_to_o0_o0_r3_c0"} : (!dataflow.bits<1>, none) -> (none, none)
    %137:2 = fabric.instance @pe_cond_br_i1_i32(%52#8, %71#6) {sym_name = "pe_cond_br_i1_i32_r2_c0"} : (!dataflow.bits<1>, !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>)
    %138 = fabric.instance @pe_constant_i0_to_o32(%6#6) {sym_name = "pe_constant_i0_to_o32_r2_c1"} : (none) -> !dataflow.bits<32>
    %139 = fabric.instance @pe_constant_i0_to_o32(%7#8) {sym_name = "pe_constant_i0_to_o32_r2_c2"} : (none) -> !dataflow.bits<32>
    %140 = fabric.instance @pe_constant_i0_to_o57(%8#8) {sym_name = "pe_constant_i0_to_o57_r1_c1"} : (none) -> !dataflow.bits<57>
    %141 = fabric.instance @pe_constant_i0_to_o57(%9#8) {sym_name = "pe_constant_i0_to_o57_r1_c2"} : (none) -> !dataflow.bits<57>
    %142 = fabric.instance @pe_constant_i0_to_o57(%10#8) {sym_name = "pe_constant_i0_to_o57_r1_c3"} : (none) -> !dataflow.bits<57>
    %143 = fabric.instance @pe_constant_i0_to_o64(%12#6) {sym_name = "pe_constant_i0_to_o64_r0_c1"} : (none) -> !dataflow.bits<64>
    %144 = fabric.instance @pe_join_i0(%13#8) {sym_name = "pe_join_i0_r2_c1"} : (none) -> none
    %145 = fabric.instance @pe_join_i0_i0(%14#8, %20#9) {sym_name = "pe_join_i0_i0_r2_c2"} : (none, none) -> none
    %146 = fabric.instance @pe_mux_i57_i0_i0_to_o0(%96#6, %15#8, %21#8) {sym_name = "pe_mux_i57_i0_i0_to_o0_r2_c0"} : (!dataflow.bits<57>, none, none) -> none
    %147 = fabric.instance @pe_mux_i57_i0_i0_to_o0(%97#8, %16#8, %22#8) {sym_name = "pe_mux_i57_i0_i0_to_o0_r2_c1"} : (!dataflow.bits<57>, none, none) -> none
    %148 = fabric.instance @pe_mux_i57_i32_i32_to_o32(%98#8, %74#8, %79#8) {sym_name = "pe_mux_i57_i32_i32_to_o32_r2_c2"} : (!dataflow.bits<57>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    fabric.instance @pe_sink_i1(%53#8) {sym_name = "pe_sink_i1_r3_c2"} : (!dataflow.bits<1>) -> ()
    %149:2 = fabric.instance @load_pe_w32(%99#8, %76#6, %18#6) {sym_name = "load_pe_r3_c0"} : (!dataflow.bits<57>, !dataflow.bits<32>, none) -> (!dataflow.bits<32>, !dataflow.bits<57>)
    %150:2 = fabric.instance @load_pe_w32f(%101#6, %77#8, %19#8) {sym_name = "load_pe_r3_c1"} : (!dataflow.bits<57>, !dataflow.bits<32>, none) -> (!dataflow.bits<32>, !dataflow.bits<57>)
    %151:2 = fabric.instance @store_pe_w32f(%102#8, %78#8, %20#8) {sym_name = "store_pe_r3_c2"} : (!dataflow.bits<57>, !dataflow.bits<32>, none) -> (!dataflow.bits<32>, !dataflow.bits<57>)
    %152 = fabric.extmemory [ldCount = 0, stCount = 1, lsqDepth = 1] (%arg0, %85#5, %110#4) : memref<?xf32, strided<[1], offset: ?>>, (memref<?xf32, strided<[1], offset: ?>>, !dataflow.bits<32>, !dataflow.bits<57>) -> (none)
    %153:2 = fabric.extmemory [ldCount = 1, stCount = 0] (%arg1, %110#5) : memref<?xi32, strided<[1], offset: ?>>, (memref<?xi32, strided<[1], offset: ?>>, !dataflow.bits<57>) -> (!dataflow.bits<32>, none)
    %154:2 = fabric.extmemory [ldCount = 1, stCount = 0] (%arg2, %110#6) : memref<?xf32, strided<[1], offset: ?>>, (memref<?xf32, strided<[1], offset: ?>>, !dataflow.bits<57>) -> (!dataflow.bits<32>, none)
    fabric.yield %35#4, %85#4 : none, !dataflow.bits<32>
  }
}
