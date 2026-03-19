module {
  handshake.func @function_unit_stream_timing_invalid(%start: index, %step: index, %bound: index)
      -> (index, i1) {
    %idx, %cont = dataflow.stream %start, %step, %bound
        {step_op = "+=", cont_cond = "<"} : (index, index, index) -> (index, i1)
    handshake.return %idx, %cont : index, i1
  }
}
