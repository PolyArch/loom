module {
  handshake.func @single_stream(%start: index, %step: index, %bound: index, ...) -> (index, i1)
      attributes {argNames = ["start", "step", "bound"], loom.annotations = ["loom.accel"],
                  resNames = ["idx", "wc"]} {
    %idx, %wc = dataflow.stream %start, %step, %bound {step_op = "+=", stop_cond = "!="}
    handshake.return %idx, %wc : index, i1
  }
}
