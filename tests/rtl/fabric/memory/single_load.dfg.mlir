// Companion DFG for single_load.fabric.mlir.
//
// ADG module boundary:
//   inputs: idx (bits<64>), ctrl (bits<64>)
//   outputs: pe_ld#0 (load data), mem0#1 (memory load_done completion)
//
// The module's second public output is the memory instance's load_done
// channel (a completion token), NOT the FU's internal forwarded address.
// Per spec-fabric-memory-interface.md, memory response families are:
// load_data, load_done, store_done.
module {
  handshake.func @mem_load_test(%idx: index, %ctrl: none,
                                 %mem: memref<256xi32>) -> (i32, none)
      attributes {argNames = ["idx", "ctrl", "mem"],
                  resNames = ["data", "load_done"]} {
    %data, %done = handshake.load [%mem] %idx, %ctrl : index, none
    handshake.return %data, %done : i32, none
  }
}
