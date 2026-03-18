module {
  fabric.spatial_pe @const_pe(%ctrl: none) -> (i32) {
    fabric.function_unit @fu_constant(%arg0: none) -> (i32)
        [latency = 1, interval = 1] {
      %0 = handshake.constant %arg0 {value = 0 : i32} : i32
      fabric.yield %0 : i32
    }
    fabric.yield
  }

  fabric.spatial_pe @cmpi_pe(%a: i32, %b: i32) -> (i1) {
    fabric.function_unit @fu_cmpi(%arg0: i32, %arg1: i32) -> (i1)
        [latency = 1, interval = 1] {
      %0 = arith.cmpi eq, %arg0, %arg1 : i32
      fabric.yield %0 : i1
    }
    fabric.yield
  }

  fabric.spatial_pe @cmpf_pe(%a: f32, %b: f32) -> (i1) {
    fabric.function_unit @fu_cmpf(%arg0: f32, %arg1: f32) -> (i1)
        [latency = 1, interval = 1] {
      %0 = arith.cmpf oeq, %arg0, %arg1 : f32
      fabric.yield %0 : i1
    }
    fabric.yield
  }

  fabric.spatial_pe @stream_pe(%start: index, %step: index, %bound: index)
      -> (index, i1) {
    fabric.function_unit @fu_stream(%arg0: index, %arg1: index, %arg2: index)
        -> (index, i1) [latency = 1, interval = 1] {
      %0, %1 = dataflow.stream %arg0, %arg1, %arg2
          {step_op = "+=", cont_cond = "<"} : (index, index, index) -> (index, i1)
      fabric.yield %0, %1 : index, i1
    }
    fabric.yield
  }

  fabric.module @function_unit_configs_test(
      %ctrl: none,
      %a: i32, %b: i32,
      %af: f32, %bf: f32,
      %start: index, %step: index, %bound: index)
      -> (i32, i1, i1, index, i1) {
    %const = fabric.instance @const_pe(%ctrl) {sym_name = "pe_const"}
        : (none) -> (i32)
    %cmpi = fabric.instance @cmpi_pe(%a, %b) {sym_name = "pe_cmpi"}
        : (i32, i32) -> (i1)
    %cmpf = fabric.instance @cmpf_pe(%af, %bf) {sym_name = "pe_cmpf"}
        : (f32, f32) -> (i1)
    %stream:2 = fabric.instance @stream_pe(%start, %step, %bound)
        {sym_name = "pe_stream"} : (index, index, index) -> (index, i1)
    fabric.yield %const, %cmpi, %cmpf, %stream#0, %stream#1
        : i32, i1, i1, index, i1
  }
}
