// Companion DFG for single_load.fabric.mlir.
// ADG boundary: 3 inputs (dram:memref<?xi32>, idx:bits<64>, ctrl:bits<64>) -> 2 outputs (bits<64>, bits<64>)
// DFG: load from external memory via extmemory interface.
module {
  handshake.func @ext_load_test(%idx: i64, %ctrl: i64, %mem: memref<?xi64>) -> (i64, i64)
      attributes {argNames = ["idx", "ctrl", "mem"], resNames = ["data", "addr_out"]} {
    %data, %addr = handshake.load [%mem] %idx, %ctrl : i64, i64, memref<?xi64> -> i64, i64
    handshake.return %data, %addr : i64, i64
  }
}
