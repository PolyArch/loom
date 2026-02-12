// RUN: loom --adg %s

// A valid load PE: body contains exactly one handshake.load and no other
// non-terminator operations.
fabric.module @test(%addr: index, %data: i32, %ctrl: none) -> (i32, index) {
  %d, %a = fabric.pe %addr, %data, %ctrl
      : (index, i32, none) -> (i32, index) {
  ^bb0(%x: index, %y: i32, %c: none):
    %ld_d, %ld_a = handshake.load [%x] %y, %c : index, i32
    fabric.yield %ld_d, %ld_a : i32, index
  }
  fabric.yield %d, %a : i32, index
}
