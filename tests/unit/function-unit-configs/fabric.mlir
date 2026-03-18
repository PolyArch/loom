module {
  fabric.spatial_sw @ctrl_fanout [connectivity_table = ["1", "1"]] : (none) -> (none, none)
  fabric.spatial_sw @a_fanout [connectivity_table = ["1", "1"]] : (i32) -> (i32, i32)

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

  fabric.spatial_pe @join_pe(%ctrl: none, %a: i32, %ctrl2: none, %guard: i1)
      -> (none) {
    fabric.function_unit @fu_join(%arg0: none, %arg1: i32, %arg2: none, %arg3: i1)
        -> (none) [latency = 1, interval = 1] {
      %0 = handshake.join %arg0, %arg1, %arg2, %arg3
          : none, i32, none, i1
      fabric.yield %0 : none
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
      %ctrl2: none,
      %a: i32, %b: i32,
      %join_guard: i1,
      %af: f32, %bf: f32,
      %start: index, %step: index, %bound: index)
      -> (i32, none, i1, i1, index, i1) {
    %ctrl_f:2 = fabric.instance @ctrl_fanout(%ctrl) {sym_name = "sw_ctrl"}
        : (none) -> (none, none)
    %a_f:2 = fabric.instance @a_fanout(%a) {sym_name = "sw_a"}
        : (i32) -> (i32, i32)
    %const = fabric.instance @const_pe(%ctrl_f#0) {sym_name = "pe_const"}
        : (none) -> (i32)
    %join = fabric.instance @join_pe(%ctrl_f#1, %a_f#0, %ctrl2, %join_guard)
        {sym_name = "pe_join"} : (none, i32, none, i1) -> (none)
    %cmpi = fabric.instance @cmpi_pe(%a_f#1, %b) {sym_name = "pe_cmpi"}
        : (i32, i32) -> (i1)
    %cmpf = fabric.instance @cmpf_pe(%af, %bf) {sym_name = "pe_cmpf"}
        : (f32, f32) -> (i1)
    %stream:2 = fabric.instance @stream_pe(%start, %step, %bound)
        {sym_name = "pe_stream"} : (index, index, index) -> (index, i1)
    fabric.yield %const, %join, %cmpi, %cmpf, %stream#0, %stream#1
        : i32, none, i1, i1, index, i1
  }
}
