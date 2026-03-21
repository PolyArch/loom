// Companion DFG for single_load.fabric.mlir.
module {
  handshake.func @mem_load_test(%addr: index, %mem: memref<?xi32>) -> (i32)
      attributes {argNames = ["addr", "mem"], resNames = ["data"]} {
    %data = handshake.load [%mem] %addr : index, memref<?xi32> -> i32
    handshake.return %data : i32
  }
}
