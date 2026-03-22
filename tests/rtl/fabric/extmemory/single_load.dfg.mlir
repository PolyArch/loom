// Companion DFG for single_load.fabric.mlir.
//
// ADG module boundary (32-bit):
//   inputs: (%dram: memref<?xi32>, %idx: bits<32>, %ctrl: bits<32>)
//   outputs: (load data bits<32>, load done bits<32>)
module {
  handshake.func @ext_load_test(%mem: memref<?xi32>, %idx: index,
                                 %data: i32, %ctrl: none) -> (i32, index)
      attributes {argNames = ["mem", "idx", "data", "ctrl"],
                  resNames = ["out_data", "out_addr"]} {
    %0, %1 = handshake.load [%idx] %data, %ctrl : index, i32
    handshake.return %0, %1 : i32, index
  }
}
