// Companion DFG for single_load.fabric.mlir.
//
// ADG module boundary:
//   inputs: (%dram: memref<?xi32>, %idx: bits<64>, %ctrl: bits<64>)
//   outputs: (pe_ld#0 = load data, ext0#1 = extmemory load_done)
//
// memref is argument 0, matching the ADG's public argument position.
// The module's second output is the extmemory instance's load_done
// completion channel, NOT the FU's internal forwarded address.
module {
  handshake.func @ext_load_test(%mem: memref<?xi32>, %idx: index, %ctrl: none)
      -> (i32, none)
      attributes {argNames = ["mem", "idx", "ctrl"],
                  resNames = ["data", "load_done"]} {
    %data, %done = handshake.load [%mem] %idx, %ctrl : index, none
    handshake.return %data, %done : i32, none
  }
}
