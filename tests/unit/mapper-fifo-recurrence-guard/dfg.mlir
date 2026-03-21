module {
  handshake.func @mapper_fifo_recurrence_guard(
      %start: index, %step: index, %bound: index, %init: index, ...) -> (index)
      attributes {
        argNames = ["start", "step", "bound", "init"],
        resNames = ["out"]
      } {
    %0, %1 = dataflow.stream %start, %step, %bound
        {step_op = "+=", cont_cond = "<"}
        : (index, index, index) -> (index, i1)
    %2, %3 = dataflow.gate %0, %1 : index, i1 -> index, i1
    %4 = dataflow.carry %3, %init, %2 : i1, index, index -> index
    return %4 : index
  }
}
