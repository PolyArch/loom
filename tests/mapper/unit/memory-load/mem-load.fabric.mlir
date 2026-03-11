module {
  fabric.pe @load_pe_w32(%arg0: !dataflow.bits<57>, %arg1: !dataflow.bits<32>, %arg2: none) [latency = [1 : i16, 1 : i16, 1 : i16], interval = [1 : i16, 1 : i16, 1 : i16]] -> (!dataflow.bits<32>, !dataflow.bits<57>) {
  ^bb0(%arg0: index, %arg1: i32, %arg2: none):
    %dataResult, %addressResults = handshake.load [%arg0] %arg1, %arg2 : index, i32
    fabric.yield %dataResult, %addressResults : i32, index
  }
  fabric.module @genadg_1(%arg0: memref<?xi32, strided<[1], offset: ?>>, %arg1: none, %arg2: !dataflow.bits<57>) -> (none, !dataflow.bits<32>) {
    %0:5 = fabric.switch [connectivity_table = ["111111", "111111", "111111", "111111", "111111"]] %1#2, %1#3, %2#0, %2#1, %arg1, %13#1 : none -> none, none, none, none, none
    %1:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %3#0, %3#1, %0#0, %0#1 : none -> none, none, none, none
    %2:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %0#2, %0#3, %3#2, %3#3 : none -> none, none, none, none
    %3:5 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111", "1111"]] %1#0, %1#1, %2#2, %2#3 : none -> none, none, none, none, none
    %4:5 = fabric.switch [connectivity_table = ["11111", "11111", "11111", "11111", "11111"]] %5#2, %5#3, %6#0, %6#1, %13#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %5:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %7#0, %7#1, %4#0, %4#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %6:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %4#2, %4#3, %7#2, %7#3 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %7:5 = fabric.switch [connectivity_table = ["11111", "11111", "11111", "11111", "11111"]] %5#0, %5#1, %6#2, %6#3, %12#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>
    %8:5 = fabric.switch [connectivity_table = ["11111", "11111", "11111", "11111", "11111"]] %9#2, %9#3, %10#0, %10#1, %arg2 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %9:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %11#0, %11#1, %8#0, %8#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %10:4 = fabric.switch [connectivity_table = ["1111", "1111", "1111", "1111"]] %8#2, %8#3, %11#2, %11#3 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %11:5 = fabric.switch [connectivity_table = ["11111", "11111", "11111", "11111", "11111"]] %9#0, %9#1, %10#2, %10#3, %12#1 : !dataflow.bits<57> -> !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>
    %12:2 = fabric.instance @load_pe_w32(%8#4, %4#4, %0#4) {sym_name = "load_pe_r0_c0"} : (!dataflow.bits<57>, !dataflow.bits<32>, none) -> (!dataflow.bits<32>, !dataflow.bits<57>)
    %13:2 = fabric.extmemory [ldCount = 1, stCount = 0] (%arg0, %11#4) : memref<?xi32, strided<[1], offset: ?>>, (memref<?xi32, strided<[1], offset: ?>>, !dataflow.bits<57>) -> (!dataflow.bits<32>, none)
    fabric.yield %3#4, %7#4 : none, !dataflow.bits<32>
  }
}
