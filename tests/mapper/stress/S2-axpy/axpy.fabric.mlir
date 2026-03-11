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
  fabric.pe @pe_muli_i32_i32(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.muli %arg0, %arg1 : i32
    fabric.yield %0 : i32
  }
  fabric.pe @pe_carry_i1_i0_i0_to_o0(%arg0: !dataflow.bits<1>, %arg1: none, %arg2: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
  ^bb0(%arg0: i1, %arg1: none, %arg2: none):
    %0 = dataflow.carry %arg0, %arg1, %arg2 : i1, none, none -> none
    fabric.yield %0 : none
  }
  fabric.pe @pe_gate_i57_i1_to_o57_o1(%arg0: !dataflow.bits<57>, %arg1: !dataflow.bits<1>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<57>, !dataflow.bits<1>) {
  ^bb0(%arg0: index, %arg1: i1):
    %afterValue, %afterCond = dataflow.gate %arg0, %arg1 : index, i1 -> index, i1
    fabric.yield %afterValue, %afterCond : index, i1
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
  fabric.pe @pe_constant_i0_to_o57(%arg0: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<57>) {
  ^bb0(%arg0: none):
    %0 = handshake.constant %arg0 {value = 0 : index} : index
    fabric.yield %0 : index
  }
  fabric.pe @pe_join_i0(%arg0: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
    %0 = handshake.join %arg0 : none
    fabric.yield %0 : none
  }
  fabric.pe @pe_join_i0_i0_i0(%arg0: none, %arg1: none, %arg2: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (none) {
    %0 = handshake.join %arg0, %arg1, %arg2 : none, none, none
    fabric.yield %0 : none
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
  fabric.module @genadg_1(%arg0: memref<?xi32, strided<[1], offset: ?>>, %arg1: memref<?xi32, strided<[1], offset: ?>>, %arg2: memref<?xi32, strided<[1], offset: ?>>, %arg3: none, %arg4: !dataflow.bits<32>, %arg5: !dataflow.bits<32>) -> (none) {
    %0:5 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111"]] %1#4, %1#5, %5#0, %5#1, %arg3, %113, %114#1, %115#1 : none -> none, none, none, none, none
    %1:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %2#4, %2#5, %6#0, %6#1, %0#0, %0#1 : none -> none, none, none, none, none, none, none
    %2:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %3#4, %3#5, %7#0, %7#1, %1#0, %1#1 : none -> none, none, none, none, none, none, none
    %3:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %4#2, %4#3, %8#0, %8#1, %2#0, %2#1 : none -> none, none, none, none, none, none, none
    %4:4 = fabric.switch [connectivity_table = ["11111", "11111", "11111", "11111"]] %9#0, %9#1, %3#0, %3#1, %103#1 : none -> none, none, none, none
    %5:8 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %0#2, %0#3, %6#6, %6#7, %10#0, %10#1 : none -> none, none, none, none, none, none, none, none
    %6:10 = fabric.switch [connectivity_table = ["1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111"]] %1#2, %1#3, %7#6, %7#7, %11#0, %11#1, %5#2, %5#3, %97, %104#1 : none -> none, none, none, none, none, none, none, none, none, none
    %7:10 = fabric.switch [connectivity_table = ["1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111", "1111111111"]] %2#2, %2#3, %8#6, %8#7, %12#0, %12#1, %6#2, %6#3, %98, %105#1 : none -> none, none, none, none, none, none, none, none, none, none
    %8:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %3#2, %3#3, %9#4, %9#5, %13#0, %13#1, %7#2, %7#3, %99 : none -> none, none, none, none, none, none, none, none, none
    %9:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %4#0, %4#1, %14#0, %14#1, %8#2, %8#3, %103#0 : none -> none, none, none, none, none, none
    %10:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %5#4, %5#5, %11#6, %11#7, %15#0, %15#1 : none -> none, none, none, none, none, none, none
    %11:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %6#4, %6#5, %12#6, %12#7, %16#0, %16#1, %10#2, %10#3, %104#0 : none -> none, none, none, none, none, none, none, none, none
    %12:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %7#4, %7#5, %13#6, %13#7, %17#0, %17#1, %11#2, %11#3, %105#0 : none -> none, none, none, none, none, none, none, none, none, none
    %13:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %8#4, %8#5, %14#4, %14#5, %18#0, %18#1, %12#2, %12#3 : none -> none, none, none, none, none, none, none, none, none
    %14:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %9#2, %9#3, %19#0, %19#1, %13#2, %13#3 : none -> none, none, none, none, none, none
    %15:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %10#4, %10#5, %16#6, %16#7, %20#0, %20#1 : none -> none, none, none, none, none, none, none
    %16:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %11#4, %11#5, %17#6, %17#7, %21#0, %21#1, %15#2, %15#3, %108 : none -> none, none, none, none, none, none, none, none, none
    %17:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %12#4, %12#5, %18#6, %18#7, %22#0, %22#1, %16#2, %16#3, %109 : none -> none, none, none, none, none, none, none, none
    %18:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %13#4, %13#5, %19#4, %19#5, %23#0, %23#1, %17#2, %17#3 : none -> none, none, none, none, none, none, none, none
    %19:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %14#2, %14#3, %24#0, %24#1, %18#2, %18#3 : none -> none, none, none, none, none, none
    %20:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %15#4, %15#5, %21#4, %21#5 : none -> none, none, none, none
    %21:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %16#4, %16#5, %22#4, %22#5, %20#2, %20#3 : none -> none, none, none, none, none, none
    %22:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %17#4, %17#5, %23#4, %23#5, %21#2, %21#3 : none -> none, none, none, none, none, none
    %23:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %18#4, %18#5, %24#2, %24#3, %22#2, %22#3 : none -> none, none, none, none, none, none
    %24:5 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111"]] %19#2, %19#3, %23#2, %23#3 : none -> none, none, none, none, none
    %25:5 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111"]] %26#4, %26#5, %29#0, %29#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %26:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %27#4, %27#5, %30#0, %30#1, %25#0, %25#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %27:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %28#2, %28#3, %31#0, %31#1, %26#0, %26#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %28:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %32#0, %32#1, %27#0, %27#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %29:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %25#2, %25#3, %30#6, %30#7, %33#0, %33#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %30:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %26#2, %26#3, %31#6, %31#7, %34#0, %34#1, %29#2, %29#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %31:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %27#2, %27#3, %32#4, %32#5, %35#0, %35#1, %30#2, %30#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %32:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %28#0, %28#1, %36#0, %36#1, %31#2, %31#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %33:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %29#4, %29#5, %34#6, %34#7, %37#0, %37#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %34:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %30#4, %30#5, %35#6, %35#7, %38#0, %38#1, %33#2, %33#3, %100#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %35:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %31#4, %31#5, %36#4, %36#5, %39#0, %39#1, %34#2, %34#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %36:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %32#2, %32#3, %40#0, %40#1, %35#2, %35#3, %102#1 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %37:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %33#4, %33#5, %38#4, %38#5 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %38:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %34#4, %34#5, %39#4, %39#5, %37#2, %37#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %39:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %35#4, %35#5, %40#2, %40#3, %38#2, %38#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %40:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %36#2, %36#3, %39#2, %39#3 : !dataflow.bits<1> -> !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>, !dataflow.bits<1>
    %41:5 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111"]] %42#4, %42#5, %45#0, %45#1, %arg4, %arg5, %114#0, %115#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %42:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %43#4, %43#5, %46#0, %46#1, %41#0, %41#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %43:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %44#2, %44#3, %47#0, %47#1, %42#0, %42#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %44:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %48#0, %48#1, %43#0, %43#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %45:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %41#2, %41#3, %46#6, %46#7, %49#0, %49#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %46:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %42#2, %42#3, %47#6, %47#7, %50#0, %50#1, %45#2, %45#3, %91 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %47:9 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %43#2, %43#3, %48#4, %48#5, %51#0, %51#1, %46#2, %46#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %48:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %44#0, %44#1, %52#0, %52#1, %47#2, %47#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %49:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %45#4, %45#5, %50#6, %50#7, %53#0, %53#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %50:10 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %46#4, %46#5, %51#6, %51#7, %54#0, %54#1, %49#2, %49#3, %94 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %51:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %47#4, %47#5, %52#4, %52#5, %55#0, %55#1, %50#2, %50#3, %96 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %52:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %48#2, %48#3, %56#0, %56#1, %51#2, %51#3, %101 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %53:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %49#4, %49#5, %54#4, %54#5 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %54:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %50#4, %50#5, %55#4, %55#5, %53#2, %53#3, %110#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %55:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %51#4, %51#5, %56#2, %56#3, %54#2, %54#3, %111#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %56:5 = fabric.switch [connectivity_table = ["11111", "11111", "11111", "11111", "11111"]] %52#2, %52#3, %55#2, %55#3, %112#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %57:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %58#4, %58#5, %62#0, %62#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %58:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %59#4, %59#5, %63#0, %63#1, %57#0, %57#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %59:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %60#4, %60#5, %64#0, %64#1, %58#0, %58#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %60:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %61#2, %61#3, %65#0, %65#1, %59#0, %59#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %61:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %66#0, %66#1, %60#0, %60#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %62:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %57#2, %57#3, %63#6, %63#7, %67#0, %67#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %63:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %58#2, %58#3, %64#6, %64#7, %68#0, %68#1, %62#2, %62#3, %93 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %64:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %59#2, %59#3, %65#6, %65#7, %69#0, %69#1, %63#2, %63#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %65:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %60#2, %60#3, %66#4, %66#5, %70#0, %70#1, %64#2, %64#3, %95 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %66:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %61#0, %61#1, %71#0, %71#1, %65#2, %65#3, %100#0 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %67:8 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %62#4, %62#5, %68#6, %68#7, %72#0, %72#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %68:9 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %63#4, %63#5, %69#6, %69#7, %73#0, %73#1, %67#2, %67#3, %102#0 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %69:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %64#4, %64#5, %70#6, %70#7, %74#0, %74#1, %68#2, %68#3, %106 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %70:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %65#4, %65#5, %71#4, %71#5, %75#0, %75#1, %69#2, %69#3, %107 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %71:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %66#2, %66#3, %76#0, %76#1, %70#2, %70#3, %110#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %72:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %67#4, %67#5, %73#6, %73#7, %77#0, %77#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %73:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %68#4, %68#5, %74#6, %74#7, %78#0, %78#1, %72#2, %72#3, %111#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %74:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %69#4, %69#5, %75#6, %75#7, %79#0, %79#1, %73#2, %73#3, %112#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %75:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %70#4, %70#5, %76#4, %76#5, %80#0, %80#1, %74#2, %74#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %76:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %71#2, %71#3, %81#0, %81#1, %75#2, %75#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %77:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %72#4, %72#5, %78#4, %78#5 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %78:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %73#4, %73#5, %79#4, %79#5, %77#2, %77#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %79:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %74#4, %74#5, %80#4, %80#5, %78#2, %78#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %80:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %75#4, %75#5, %81#2, %81#3, %79#2, %79#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %81:7 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111", "1111", "1111"]] %76#2, %76#3, %80#2, %80#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %82:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %83#4, %83#5, %85#0, %85#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %83:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %84#2, %84#3, %86#0, %86#1, %82#0, %82#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %84:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %87#0, %87#1, %83#0, %83#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %85:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %82#2, %82#3, %86#6, %86#7, %88#0, %88#1 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %86:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %83#2, %83#3, %87#4, %87#5, %89#0, %89#1, %85#2, %85#3, %92 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %87:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %84#0, %84#1, %90#0, %90#1, %86#2, %86#3 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %88:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %85#4, %85#5, %89#4, %89#5 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %89:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %86#4, %86#5, %90#2, %90#3, %88#2, %88#3 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %90:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %87#2, %87#3, %89#2, %89#3 : !dataflow.bits<64> -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>
    %91 = fabric.instance @pe_addi_i32_i32(%41#4, %45#6) {sym_name = "pe_addi_i32_i32_r0_c0"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %92 = fabric.instance @pe_extui_i32_to_o64(%42#6) {sym_name = "pe_extui_i32_to_o64_r0_c1"} : (!dataflow.bits<32>) -> !dataflow.bits<64>
    %93 = fabric.instance @pe_index_cast_i32_to_o57(%43#6) {sym_name = "pe_index_cast_i32_to_o57_r0_c2"} : (!dataflow.bits<32>) -> !dataflow.bits<57>
    %94 = fabric.instance @pe_index_cast_i57_to_o32(%58#6) {sym_name = "pe_index_cast_i57_to_o32_r0_c1"} : (!dataflow.bits<57>) -> !dataflow.bits<32>
    %95 = fabric.instance @pe_index_cast_i64_to_o57(%83#6) {sym_name = "pe_index_cast_i64_to_o57_r0_c1"} : (!dataflow.bits<64>) -> !dataflow.bits<57>
    %96 = fabric.instance @pe_muli_i32_i32(%46#8, %50#9) {sym_name = "pe_muli_i32_i32_r1_c1"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %97 = fabric.instance @pe_carry_i1_i0_i0_to_o0(%25#4, %0#4, %5#7) {sym_name = "pe_carry_i1_i0_i0_to_o0_r0_c0"} : (!dataflow.bits<1>, none, none) -> none
    %98 = fabric.instance @pe_carry_i1_i0_i0_to_o0(%26#6, %1#6, %6#9) {sym_name = "pe_carry_i1_i0_i0_to_o0_r0_c1"} : (!dataflow.bits<1>, none, none) -> none
    %99 = fabric.instance @pe_carry_i1_i0_i0_to_o0(%27#6, %2#6, %7#9) {sym_name = "pe_carry_i1_i0_i0_to_o0_r0_c2"} : (!dataflow.bits<1>, none, none) -> none
    %100:2 = fabric.instance @pe_gate_i57_i1_to_o57_o1(%60#6, %29#6) {sym_name = "pe_gate_i57_i1_to_o57_o1_r0_c3"} : (!dataflow.bits<57>, !dataflow.bits<1>) -> (!dataflow.bits<57>, !dataflow.bits<1>)
    %101 = fabric.instance @pe_invariant_i1_i32(%30#8, %47#8) {sym_name = "pe_invariant_i1_i32_r1_c2"} : (!dataflow.bits<1>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %102:2 = fabric.instance @pe_stream_i57_i57_i57_to_o57_o1(%62#6, %67#7, %63#8) {sym_name = "pe_stream_i57_i57_i57_to_o57_o1_r1_c0"} : (!dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>) -> (!dataflow.bits<57>, !dataflow.bits<1>)
    %103:2 = fabric.instance @pe_cond_br_i1_i0_to_o0_o0(%33#6, %3#6) {sym_name = "pe_cond_br_i1_i0_to_o0_o0_r2_c0"} : (!dataflow.bits<1>, none) -> (none, none)
    %104:2 = fabric.instance @pe_cond_br_i1_i0_to_o0_o0(%34#8, %5#6) {sym_name = "pe_cond_br_i1_i0_to_o0_o0_r2_c1"} : (!dataflow.bits<1>, none) -> (none, none)
    %105:2 = fabric.instance @pe_cond_br_i1_i0_to_o0_o0(%35#8, %6#8) {sym_name = "pe_cond_br_i1_i0_to_o0_o0_r2_c2"} : (!dataflow.bits<1>, none) -> (none, none)
    %106 = fabric.instance @pe_constant_i0_to_o57(%7#8) {sym_name = "pe_constant_i0_to_o57_r1_c1"} : (none) -> !dataflow.bits<57>
    %107 = fabric.instance @pe_constant_i0_to_o57(%8#8) {sym_name = "pe_constant_i0_to_o57_r1_c2"} : (none) -> !dataflow.bits<57>
    %108 = fabric.instance @pe_join_i0(%10#6) {sym_name = "pe_join_i0_r2_c0"} : (none) -> none
    %109 = fabric.instance @pe_join_i0_i0_i0(%11#8, %16#8, %12#9) {sym_name = "pe_join_i0_i0_i0_r2_c1"} : (none, none, none) -> none
    %110:2 = fabric.instance @load_pe_w32(%65#8, %49#6, %12#8) {sym_name = "load_pe_r2_c0"} : (!dataflow.bits<57>, !dataflow.bits<32>, none) -> (!dataflow.bits<32>, !dataflow.bits<57>)
    %111:2 = fabric.instance @load_pe_w32(%67#6, %50#8, %13#8) {sym_name = "load_pe_r2_c1"} : (!dataflow.bits<57>, !dataflow.bits<32>, none) -> (!dataflow.bits<32>, !dataflow.bits<57>)
    %112:2 = fabric.instance @store_pe_w32(%68#8, %51#8, %15#6) {sym_name = "store_pe_r2_c2"} : (!dataflow.bits<57>, !dataflow.bits<32>, none) -> (!dataflow.bits<32>, !dataflow.bits<57>)
    %113 = fabric.extmemory [ldCount = 0, stCount = 1, lsqDepth = 1] (%arg0, %56#4, %81#4) : memref<?xi32, strided<[1], offset: ?>>, (memref<?xi32, strided<[1], offset: ?>>, !dataflow.bits<32>, !dataflow.bits<57>) -> (none)
    %114:2 = fabric.extmemory [ldCount = 1, stCount = 0] (%arg1, %81#5) : memref<?xi32, strided<[1], offset: ?>>, (memref<?xi32, strided<[1], offset: ?>>, !dataflow.bits<57>) -> (!dataflow.bits<32>, none)
    %115:2 = fabric.extmemory [ldCount = 1, stCount = 0] (%arg2, %81#6) : memref<?xi32, strided<[1], offset: ?>>, (memref<?xi32, strided<[1], offset: ?>>, !dataflow.bits<57>) -> (!dataflow.bits<32>, none)
    fabric.yield %24#4 : none
  }
}
