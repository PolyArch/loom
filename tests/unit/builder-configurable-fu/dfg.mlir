module {
  handshake.func @builder_configurable_fu(
      %ctrl: none,
      %ctrl2: none,
      %a: i32, %b: i32,
      %join_guard: i1,
      %af: f32, %bf: f32,
      %start: index, %step: index, %bound: index, ...)
      -> (i32, none, i1, i1, index, i1)
      attributes {
        argNames = ["ctrl", "ctrl2", "a", "b", "join_guard", "af", "bf", "start", "step", "bound"],
        resNames = ["const", "join_done", "cmpi", "cmpf", "stream_idx", "stream_cond"]
      } {
    %const = handshake.constant %ctrl {value = 42 : i32} : i32
    %done = handshake.join %ctrl, %a, %ctrl2 : none, i32, none
    %cmpi = arith.cmpi sgt, %a, %b : i32
    %cmpf = arith.cmpf oge, %af, %bf : f32
    %idx, %cont = dataflow.stream %start, %step, %bound
        {step_op = "+=", cont_cond = "!="}
        : (index, index, index) -> (index, i1)
    return %const, %done, %cmpi, %cmpf, %idx, %cont
        : i32, none, i1, i1, index, i1
  }
}
