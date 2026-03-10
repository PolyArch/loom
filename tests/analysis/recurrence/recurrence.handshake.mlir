module {
  handshake.func @recurrence(%start: index, %step: index, %bound: index,
                              %init: i32, ...) -> (i32)
      attributes {argNames = ["start", "step", "bound", "init"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["result"]} {
    // Loop with carry-based recurrence: acc = acc + acc * 2
    %idx, %wc = dataflow.stream %start, %step, %bound
        {loom.annotations = ["loom.loop.tripcount typical=64"]}
    %av, %ac = dataflow.gate %idx, %wc : index, i1 -> index, i1
    %val = dataflow.carry %ac, %init, %next : i1, i32, i32 -> i32

    // Recurrence chain: val -> muli -> addi -> next (feeds back to carry)
    %c2 = arith.constant 2 : i32
    %doubled = arith.muli %val, %c2 : i32
    %next = arith.addi %val, %doubled : i32

    %cb_true, %cb_false = handshake.cond_br %ac, %next : i32
    handshake.return %cb_false : i32
  }
}
