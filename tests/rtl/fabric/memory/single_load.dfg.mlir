// Companion DFG for single_load.fabric.mlir.
// ADG boundary: 2 inputs (idx:bits<64>, ctrl:bits<64>) -> 2 outputs (bits<64>, bits<64>)
// Observable: output0 = loaded data from scratchpad, output1 = memory second channel
// The DFG models the software-visible load operation.
module {
  handshake.func @mem_load_test(%idx: index, %ctrl: none,
                                 %mem: memref<256xi32>) -> (i32, index)
      attributes {argNames = ["idx", "ctrl", "mem"],
                  resNames = ["data", "mem_chan1"]} {
    %data, %addr_fwd = handshake.load [%mem] %idx, %ctrl : index, none
    handshake.return %data, %addr_fwd : i32, index
  }
}
