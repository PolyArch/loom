module {
  handshake.func @loop_compute(%start: index, %step: index, %bound: index,
                               %init: i32, ...) -> (i32)
      attributes {argNames = ["start", "step", "bound", "init"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["result"]} {
    %idx, %wc = dataflow.stream %start, %step, %bound
        {loom.annotations = ["loom.loop.tripcount typical=256"]}
    %av, %ac = dataflow.gate %idx, %wc : index, i1 -> index, i1
    %val = dataflow.carry %ac, %init, %next : i1, i32, i32 -> i32
    %one = arith.constant 1 : i32
    %next = arith.addi %val, %one : i32
    %cb_true, %cb_false = handshake.cond_br %ac, %next : i32
    handshake.return %cb_false : i32
  }
}
