// Companion DFG for single_load.fabric.mlir (external DRAM-style memory).
//
// ADG module boundary:
//   inputs:  %dram (memref<?xi32>, invisible), %idx (bits<32>), %ctrl (bits<32>)
//   outputs: data_out (bits<32>), completion (bits<32>)
//
// The DFG uses handshake.load + handshake.extmemory to model the
// external memory access through the memref argument.
module {
  handshake.func @ext_load_test(
      %mem: memref<?xi32>, %idx: index, %ctrl: none, ...)
      -> (i32, none)
      attributes {argNames = ["mem", "idx", "ctrl"],
                  resNames = ["out_data", "completion"]} {
    %lddata, %ldaddr = load [%idx] %memif#0, %ctrl : index, i32
    %memif:2 = extmemory[ld = 1, st = 0]
        (%mem : memref<?xi32>) (%ldaddr)
        {id = 0 : i32} : (index) -> (i32, none)
    return %lddata, %memif#1 : i32, none
  }
}
