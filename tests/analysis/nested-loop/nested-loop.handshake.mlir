module {
  handshake.func @nested_loop(%ostart: index, %ostep: index, %obound: index,
                               %istep: index, %ibound: index,
                               %init: i32, ...) -> (i32)
      attributes {argNames = ["ostart", "ostep", "obound",
                               "istep", "ibound", "init"],
                  loom.annotations = ["loom.accel"],
                  resNames = ["result"]} {
    // Outer loop: stream with tripcount=10
    %oidx, %owc = dataflow.stream %ostart, %ostep, %obound
        {loom.annotations = ["loom.loop.tripcount typical=10"]}
    %oav, %oac = dataflow.gate %oidx, %owc : index, i1 -> index, i1
    %oval = dataflow.carry %oac, %init, %onext : i1, i32, i32 -> i32

    // Inner loop: stream dependent on outer index (makes it truly nested).
    // Inner start = outer afterValue (index), so inner stream is inside outer body.
    %iidx, %iwc = dataflow.stream %oav, %istep, %ibound
        {loom.annotations = ["loom.loop.tripcount typical=100"]}
    %iav, %iac = dataflow.gate %iidx, %iwc : index, i1 -> index, i1
    %ival = dataflow.carry %iac, %oval, %inext : i1, i32, i32 -> i32

    // Inner loop body: add 1 to accumulator
    %c1 = arith.constant 1 : i32
    %inext = arith.addi %ival, %c1 : i32

    // Inner loop exit
    %icb_true, %icb_false = handshake.cond_br %iac, %inext : i32

    // Outer loop body: multiply by 2
    %c2 = arith.constant 2 : i32
    %onext = arith.muli %icb_false, %c2 : i32

    // Outer loop exit
    %ocb_true, %ocb_false = handshake.cond_br %oac, %onext : i32

    handshake.return %ocb_false : i32
  }
}
