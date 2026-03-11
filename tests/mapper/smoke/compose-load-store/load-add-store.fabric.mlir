module {
  fabric.pe @pe_addi_i32_i32(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.addi %arg0, %arg1 : i32
    fabric.yield %0 : i32
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
  fabric.module @genadg_1(%arg0: memref<?xi32, strided<[1], offset: ?>>, %arg1: none, %arg2: !dataflow.bits<32>, %arg3: !dataflow.bits<57>) -> (none) {
    %0:5 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111"]] %1#4, %1#5, %3#0, %3#1, %arg1, %30#1, %30#2 : none -> none, none, none, none, none
    %1:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %2#2, %2#3, %4#0, %4#1, %0#0, %0#1 : none -> none, none, none, none, none, none, none
    %2:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %5#0, %5#1, %1#0, %1#1 : none -> none, none, none, none
    %3:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %0#2, %0#3, %4#6, %4#7, %6#0, %6#1 : none -> none, none, none, none, none, none
    %4:8 = fabric.switch [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] %1#2, %1#3, %5#4, %5#5, %7#0, %7#1, %3#2, %3#3 : none -> none, none, none, none, none, none, none, none
    %5:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %2#0, %2#1, %8#0, %8#1, %4#2, %4#3 : none -> none, none, none, none, none, none
    %6:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %3#4, %3#5, %7#4, %7#5 : none -> none, none, none, none
    %7:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %4#4, %4#5, %8#2, %8#3, %6#2, %6#3 : none -> none, none, none, none, none, none
    %8:5 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111"]] %5#2, %5#3, %7#2, %7#3 : none -> none, none, none, none, none
    %9:5 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111"]] %10#4, %10#5, %12#0, %12#1, %arg2, %30#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %10:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %11#2, %11#3, %13#0, %13#1, %9#0, %9#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %11:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %14#0, %14#1, %10#0, %10#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %12:8 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %9#2, %9#3, %13#6, %13#7, %15#0, %15#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %13:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %10#2, %10#3, %14#4, %14#5, %16#0, %16#1, %12#2, %12#3, %27 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %14:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %11#0, %11#1, %17#0, %17#1, %13#2, %13#3, %28#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %15:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %12#4, %12#5, %16#4, %16#5 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %16:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %13#4, %13#5, %17#2, %17#3, %15#2, %15#3, %29#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %17:5 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111"]] %14#2, %14#3, %16#2, %16#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %18:5 = fabric.switch [connectivity_table = ["11111", "11111", "11111", "11111", "11111"]] %19#4, %19#5, %21#0, %21#1, %arg3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %19:7 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111", "111111"]] %20#2, %20#3, %22#0, %22#1, %18#0, %18#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %20:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %23#0, %23#1, %19#0, %19#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %21:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %18#2, %18#3, %22#6, %22#7, %24#0, %24#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %22:8 = fabric.switch [connectivity_table = ["111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111", "111111111"]] %19#2, %19#3, %23#4, %23#5, %25#0, %25#1, %21#2, %21#3, %28#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %23:6 = fabric.switch [connectivity_table = ["1111111", "1111111", "1111111", "1111111", "1111111", "1111111"]] %20#0, %20#1, %26#0, %26#1, %22#2, %22#3, %29#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %24:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %21#4, %21#5, %25#4, %25#5 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %25:6 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111", "111111"]] %22#4, %22#5, %26#2, %26#3, %24#2, %24#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %26:6 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111", "1111"]] %23#2, %23#3, %25#2, %25#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %27 = fabric.instance @pe_addi_i32_i32(%9#4, %12#7) {sym_name = "pe_addi_i32_i32_r0_c0"} : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    %28:2 = fabric.instance @load_pe_w32(%18#4, %10#6, %0#4) {sym_name = "load_pe_r0_c1"} : (!dataflow.bits<57>, !dataflow.bits<32>, none) -> (!dataflow.bits<32>, !dataflow.bits<57>)
    %29:2 = fabric.instance @store_pe_w32(%19#6, %12#6, %1#6) {sym_name = "store_pe_r1_c0"} : (!dataflow.bits<57>, !dataflow.bits<32>, none) -> (!dataflow.bits<32>, !dataflow.bits<57>)
    %30:3 = fabric.extmemory [ldCount = 1, stCount = 1, lsqDepth = 1] (%arg0, %17#4, %26#4, %26#5) : memref<?xi32, strided<[1], offset: ?>>, (memref<?xi32, strided<[1], offset: ?>>, !dataflow.bits<32>, !dataflow.bits<57>, !dataflow.bits<57>) -> (!dataflow.bits<32>, none, none)
    fabric.yield %8#4 : none
  }
}
