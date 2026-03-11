module {
  handshake.func @single_bitreverse(%a: i32, ...) -> (i32)
      attributes {argNames = ["a"], loom.annotations = ["loom.accel"],
                  resNames = ["out"]} {
    %0 = llvm.intr.bitreverse(%a) : (i32) -> i32
    handshake.return %0 : i32
  }
}
