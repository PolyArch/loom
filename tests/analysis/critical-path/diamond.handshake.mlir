module {
  handshake.func @diamond(%a: i32, %b: i32, %c: i32, ...) -> (i32)
      attributes {argNames = ["a", "b", "c"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["result"]} {
    // Diamond pattern: a branches into two paths of different length
    // Path 1 (short): x = a + b                      (1 op, latency 1)
    // Path 2 (long):  y = (a * c) + c  = muli + addi (2 ops, latency 2)
    // Join: z = x + y
    // Critical path should be: a -> muli -> addi_mid -> addi_join

    %x = arith.addi %a, %b : i32
    %y_mul = arith.muli %a, %c : i32
    %y = arith.addi %y_mul, %c : i32
    %z = arith.addi %x, %y : i32
    handshake.return %z : i32
  }
}
