module {
  handshake.func @function_unit_configs(
      %ctrl: none,
      %a: i32, %b: i32,
      %af: f32, %bf: f32,
      %start: index, %step: index, %bound: index, ...)
      -> (i32, i1, i1, index, i1)
      attributes {
        argNames = ["ctrl", "a", "b", "af", "bf", "start", "step", "bound"],
        resNames = ["const", "cmpi", "cmpf", "stream_idx", "stream_cond"]
      } {
    %const = handshake.constant %ctrl {value = 42 : i32} : i32
    %cmpi = arith.cmpi sgt, %a, %b : i32
    %cmpf = arith.cmpf oge, %af, %bf : f32
    %idx, %cont = dataflow.stream %start, %step, %bound
        {step_op = "+=", cont_cond = "!="}
        : (index, index, index) -> (index, i1)
    return %const, %cmpi, %cmpf, %idx, %cont : i32, i1, i1, index, i1
  }
}
