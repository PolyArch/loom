// Minimal 2-kernel TDG for integration testing.
// Producer generates a stream of values; consumer accumulates them.

module @simple_2kernel {

  // Kernel: producer -- outputs a stream of integers [0..N)
  func.func @producer(%out: memref<8xi32>, %n: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n_idx = arith.index_cast %n : i32 to index
    scf.for %i = %c0 to %n_idx step %c1 {
      %val = arith.index_cast %i : index to i32
      memref.store %val, %out[%i] : memref<8xi32>
    }
    return
  }

  // Kernel: consumer -- reads stream and accumulates
  func.func @consumer(%in: memref<8xi32>, %n: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %n_idx = arith.index_cast %n : i32 to index
    %result = scf.for %i = %c0 to %n_idx step %c1
        iter_args(%acc = %c0_i32) -> i32 {
      %val = memref.load %in[%i] : memref<8xi32>
      %new_acc = arith.addi %acc, %val : i32
      scf.yield %new_acc : i32
    }
    return %result : i32
  }

  // Contract edge: producer.out -> consumer.in
  // Attributes: ordering=FIFO, dataType=i32
}
