// Companion DFG for single_load.fabric.mlir.
// ADG boundary: (%dram: memref<?xi32>, %idx: bits<64>, %ctrl: bits<64>) -> (bits<64>, bits<64>)
// memref is argument 0, matching the ADG's public argument position.
// Observable: output0 = loaded data, output1 = extmemory second channel
module {
  handshake.func @ext_load_test(%mem: memref<?xi32>, %idx: index, %ctrl: none)
      -> (i32, index)
      attributes {argNames = ["mem", "idx", "ctrl"],
                  resNames = ["data", "ext_chan1"]} {
    %data, %addr_fwd = handshake.load [%mem] %idx, %ctrl : index, none
    handshake.return %data, %addr_fwd : i32, index
  }
}
