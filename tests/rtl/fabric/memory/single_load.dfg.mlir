// Companion DFG for single_load.fabric.mlir.
//
// ADG module boundary (visible, non-memref ports):
//   inputs:  idx (bits<32>), ctrl (bits<32>)
//   outputs: data_out (bits<32>), completion (bits<32>)
//
// The memory interface is internal to the ADG; the DFG models
// the load operation using handshake.load + handshake.memory.
module {
  handshake.func @mem_load_test(
      %idx: index, %ctrl: none, ...)
      -> (i32, none)
      attributes {argNames = ["idx", "ctrl"],
                  resNames = ["out_data", "completion"]} {
    %lddata, %ldaddr = load [%idx] %memif#0, %ctrl : index, i32
    %memif:2 = memory[ld = 1, st = 0]
        (%ldaddr) {id = 0 : i32, lsq = false}
        : memref<256xi32>, (index) -> (i32, none)
    return %lddata, %memif#1 : i32, none
  }
}
