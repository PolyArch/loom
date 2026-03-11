module {
  fabric.pe @pe_addi_i32_i32(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.addi %arg0, %arg1 : i32
    fabric.yield %0 : i32
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
  fabric.pe @pe_constant_i0_to_o57(%arg0: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<57>) {
  ^bb0(%arg0: none):
    %0 = handshake.constant %arg0 {value = 0 : index} : index
    fabric.yield %0 : index
  }
  fabric.pe @pe_join_i0(%arg0: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
    %0 = handshake.join %arg0 : none
    fabric.yield %0 : none
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
  fabric.module @genadg_1(%arg0: memref<?xi32, strided<[1], offset: ?>>, %arg1: none, %arg2: !dataflow.bits<32>, %arg3: !dataflow.bits<32>) -> (none, !dataflow.bits<32>) {
    %0:5 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111"]] %1#4, %1#5, %4#0, %4#1, %arg1, %89#1 : none -> none, none, none, none, none
    %1:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %2#4, %2#5, %5#0, %5#1, %0#0, %0#1 : none -> none, none, none, none, none, none, none
    %2:7 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %3#2, %3#3, %6#0, %6#1, %1#0, %1#1, %83#1 : none -> none, none, none, none, none, none, none
    %3:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %7#0, %7#1, %2#0, %2#1 : none -> none, none, none, none
    %4:8 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %0#2, %0#3, %5#6, %5#7, %8#0, %8#1 : none -> none, none, none, none, none, none, none, none
    %5:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %1#2, %1#3, %6#6, %6#7, %9#0, %9#1, %4#2, %4#3, %78 : none -> none, none, none, none, none, none, none, none, none
    %6:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %2#2, %2#3, %7#4, %7#5, %10#0, %10#1, %5#2, %5#3, %83#0 : none -> none, none, none, none, none, none, none, none, none
    %7:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %3#0, %3#1, %11#0, %11#1, %6#2, %6#3 : none -> none, none, none, none, none, none
    %8:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %4#4, %4#5, %9#6, %9#7, %12#0, %12#1 : none -> none, none, none, none, none, none
    %9:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %5#4, %5#5, %10#6, %10#7, %13#0, %13#1, %8#2, %8#3 : none -> none, none, none, none, none, none, none, none
    %10:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %6#4, %6#5, %11#4, %11#5, %14#0, %14#1, %9#2, %9#3, %87 : none -> none, none, none, none, none, none, none, none
    %11:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %7#2, %7#3, %15#0, %15#1, %10#2, %10#3 : none -> none, none, none, none, none, none
    %12:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %8#4, %8#5, %13#4, %13#5 : none -> none, none, none, none
    %13:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %9#4, %9#5, %14#4, %14#5, %12#2, %12#3 : none -> none, none, none, none, none, none
    %14:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %10#4, %10#5, %15#2, %15#3, %13#2, %13#3 : none -> none, none, none, none, none, none
    %15:5 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111"]] %11#2, %11#3, %14#2, %14#3 : none -> none, none, none, none, none
    %16:5 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111"]] %17#4, %17#5, %20#0, %20#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %17:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %18#4, %18#5, %21#0, %21#1, %16#0, %16#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %18:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %19#2, %19#3, %22#0, %22#1, %17#0, %17#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %19:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %23#0, %23#1, %18#0, %18#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %20:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %16#2, %16#3, %21#6, %21#7, %24#0, %24#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %21:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %17#2, %17#3, %22#6, %22#7, %25#0, %25#1, %20#2, %20#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %22:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %18#2, %18#3, %23#4, %23#5, %26#0, %26#1, %21#2, %21#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %23:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %19#0, %19#1, %27#0, %27#1, %22#2, %22#3, %80#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %24:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %20#4, %20#5, %25#6, %25#7, %28#0, %28#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %25:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %21#4, %21#5, %26#6, %26#7, %29#0, %29#1, %24#2, %24#3, %81#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %26:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %22#4, %22#5, %27#4, %27#5, %30#0, %30#1, %25#2, %25#3, %82#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %27:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %23#2, %23#3, %31#0, %31#1, %26#2, %26#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %28:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %24#4, %24#5, %29#4, %29#5 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %29:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %25#4, %25#5, %30#4, %30#5, %28#2, %28#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %30:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %26#4, %26#5, %31#2, %31#3, %29#2, %29#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %31:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %27#2, %27#3, %30#2, %30#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %32:5 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111"]] %33#4, %33#5, %36#0, %36#1, %arg2, %arg3, %89#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %33:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %34#4, %34#5, %37#0, %37#1, %32#0, %32#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %34:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %35#2, %35#3, %38#0, %38#1, %33#0, %33#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %35:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %39#0, %39#1, %34#0, %34#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %36:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %32#2, %32#3, %37#6, %37#7, %40#0, %40#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %37:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %33#2, %33#3, %38#6, %38#7, %41#0, %41#1, %36#2, %36#3, %73 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %38:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %34#2, %34#3, %39#4, %39#5, %42#0, %42#1, %37#2, %37#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %39:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %35#0, %35#1, %43#0, %43#1, %38#2, %38#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %40:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %36#4, %36#5, %41#6, %41#7, %44#0, %44#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %41:10 = fabric.switch [connectivity_table = ["1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111"]] %37#4, %37#5, %42#6, %42#7, %45#0, %45#1, %40#2, %40#3, %76, %84#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %42:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %38#4, %38#5, %43#4, %43#5, %46#0, %46#1, %41#2, %41#3, %79 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %43:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %39#2, %39#3, %47#0, %47#1, %42#2, %42#3, %80#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %44:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %40#4, %40#5, %45#4, %45#5 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %45:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %41#4, %41#5, %46#4, %46#5, %44#2, %44#3, %84#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %46:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %42#4, %42#5, %47#2, %47#3, %45#2, %45#3, %88#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %47:5 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111"]] %43#2, %43#3, %46#2, %46#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %48:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %49#4, %49#5, %52#0, %52#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %49:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %50#4, %50#5, %53#0, %53#1, %48#0, %48#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %50:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %51#2, %51#3, %54#0, %54#1, %49#0, %49#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %51:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %55#0, %55#1, %50#0, %50#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %52:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %48#2, %48#3, %53#6, %53#7, %56#0, %56#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %53:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %49#2, %49#3, %54#6, %54#7, %57#0, %57#1, %52#2, %52#3, %75 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %54:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %50#2, %50#3, %55#4, %55#5, %58#0, %58#1, %53#2, %53#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %55:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %51#0, %51#1, %59#0, %59#1, %54#2, %54#3, %77 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %56:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %52#4, %52#5, %57#6, %57#7, %60#0, %60#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %57:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %53#4, %53#5, %58#6, %58#7, %61#0, %61#1, %56#2, %56#3, %81#0 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %58:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %54#4, %54#5, %59#4, %59#5, %62#0, %62#1, %57#2, %57#3, %82#0 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %59:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %55#2, %55#3, %63#0, %63#1, %58#2, %58#3, %85 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %60:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %56#4, %56#5, %61#4, %61#5 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %61:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %57#4, %57#5, %62#4, %62#5, %60#2, %60#3, %86 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %62:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %58#4, %58#5, %63#2, %63#3, %61#2, %61#3, %88#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %63:5 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111"]] %59#2, %59#3, %62#2, %62#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %64:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %65#4, %65#5, %67#0, %67#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %65:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %66#2, %66#3, %68#0, %68#1, %64#0, %64#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %66:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %69#0, %69#1, %65#0, %65#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %67:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %64#2, %64#3, %68#6, %68#7, %70#0, %70#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %68:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %65#2, %65#3, %69#4, %69#5, %71#0, %71#1, %67#2, %67#3, %74 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %69:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %66#0, %66#1, %72#0, %72#1, %68#2, %68#3 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %70:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %67#4, %67#5, %71#4, %71#5 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %71:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %68#4, %68#5, %72#2, %72#3, %70#2, %70#3 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %72:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %69#2, %69#3, %71#2, %71#3 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %73 = fabric.instance @pe_addi_i32_i32(%32#4, %36#6) {sym_name = "pe_addi_i32_i32_r0_c0"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %74 = fabric.instance @pe_extui_i32_to_o64(%33#6) {sym_name = "pe_extui_i32_to_o64_r0_c1"} : (!dataflow.bits<32>) -> !dataflow.bits<64>
    %75 = fabric.instance @pe_index_cast_i32_to_o57(%34#6) {sym_name = "pe_index_cast_i32_to_o57_r0_c2"} : (!dataflow.bits<32>) -> !dataflow.bits<57>
    %76 = fabric.instance @pe_index_cast_i57_to_o32(%49#6) {sym_name = "pe_index_cast_i57_to_o32_r0_c1"} : (!dataflow.bits<57>) -> !dataflow.bits<32>
    %77 = fabric.instance @pe_index_cast_i64_to_o57(%65#6) {sym_name = "pe_index_cast_i64_to_o57_r0_c1"} : (!dataflow.bits<64>) -> !dataflow.bits<57>
    %78 = fabric.instance @pe_carry_i1_i0_i0_to_o0(%16#4, %0#4, %4#7) {sym_name = "pe_carry_i1_i0_i0_to_o0_r0_c0"} : (!dataflow.bits<1>, none, none) -> none
    %79 = fabric.instance @pe_carry_i1_i32_i32(%17#6, %37#8, %41#9) {sym_name = "pe_carry_i1_i32_i32_r1_c1"} : (!dataflow.bits<1>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %80:2 = fabric.instance @pe_gate_i32_i1_to_o32_o1(%38#8, %18#6) {sym_name = "pe_gate_i32_i1_to_o32_o1_r1_c2"} : (!dataflow.bits<32>, !dataflow.bits<1>) -> (!dataflow.bits<32>, !dataflow.bits<1>)
    %81:2 = fabric.instance @pe_gate_i57_i1_to_o57_o1(%52#6, %20#6) {sym_name = "pe_gate_i57_i1_to_o57_o1_r1_c0"} : (!dataflow.bits<57>, !dataflow.bits<1>) -> (!dataflow.bits<57>, !dataflow.bits<1>)
    %82:2 = fabric.instance @pe_stream_i57_i57_i57_to_o57_o1(%53#8, %57#9, %54#8) {sym_name = "pe_stream_i57_i57_i57_to_o57_o1_r1_c1"} : (!dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>) -> (!dataflow.bits<57>, !dataflow.bits<1>)
    %83:2 = fabric.instance @pe_cond_br_i1_i0_to_o0_o0(%22#8, %1#6) {sym_name = "pe_cond_br_i1_i0_to_o0_o0_r1_c2"} : (!dataflow.bits<1>, none) -> (none, none)
    %84:2 = fabric.instance @pe_cond_br_i1_i32(%24#6, %40#6) {sym_name = "pe_cond_br_i1_i32_r2_c0"} : (!dataflow.bits<1>, !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>)
    %85 = fabric.instance @pe_constant_i0_to_o57(%2#6) {sym_name = "pe_constant_i0_to_o57_r1_c2"} : (none) -> !dataflow.bits<57>
    %86 = fabric.instance @pe_constant_i0_to_o57(%4#6) {sym_name = "pe_constant_i0_to_o57_r2_c0"} : (none) -> !dataflow.bits<57>
    %87 = fabric.instance @pe_join_i0(%5#8) {sym_name = "pe_join_i0_r1_c1"} : (none) -> none
    fabric.instance @pe_sink_i1(%25#8) {sym_name = "pe_sink_i1_r2_c1"} : (!dataflow.bits<1>) -> ()
    %88:2 = fabric.instance @load_pe_w32(%57#8, %41#8, %6#8) {sym_name = "load_pe_r2_c1"} : (!dataflow.bits<57>, !dataflow.bits<32>, none) -> (!dataflow.bits<32>, !dataflow.bits<57>)
    %89:2 = fabric.extmemory [ldCount = 1, stCount = 0] (%arg0, %63#4) : memref<?xi32, strided<[1], offset: ?>>, (memref<?xi32, strided<[1], offset: ?>>, !dataflow.bits<57>) -> (!dataflow.bits<32>, none)
    fabric.yield %15#4, %47#4 : none, !dataflow.bits<32>
  }
}
