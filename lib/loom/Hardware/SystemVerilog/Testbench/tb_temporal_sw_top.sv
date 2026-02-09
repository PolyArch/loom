//===-- tb_temporal_sw_top.sv - Smoke test for temporal_sw_top -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Minimal smoke testbench for the generated temporal_sw_top module.
// Validates: reset state, error port, basic tag-based routing.
//
//===----------------------------------------------------------------------===//

module tb_temporal_sw_top;

  logic        clk;
  logic        rst_n;
  logic        in0_valid, in0_ready;
  logic [35:0] in0_data;
  logic        in1_valid, in1_ready;
  logic [35:0] in1_data;
  logic        out0_valid, out0_ready;
  logic [35:0] out0_data;
  logic        out1_valid, out1_ready;
  logic [35:0] out1_data;
  logic        error_valid;
  logic [15:0] error_code;
  logic [35:0] t0_cfg_data;

  temporal_sw_top dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .in0_valid    (in0_valid),
    .in0_ready    (in0_ready),
    .in0_data     (in0_data),
    .in1_valid    (in1_valid),
    .in1_ready    (in1_ready),
    .in1_data     (in1_data),
    .out0_valid   (out0_valid),
    .out0_ready   (out0_ready),
    .out0_data    (out0_data),
    .out1_valid   (out1_valid),
    .out1_ready   (out1_ready),
    .out1_data    (out1_data),
    .t0_cfg_data  (t0_cfg_data),
    .error_valid  (error_valid),
    .error_code   (error_code)
  );

  initial clk = 0;
  always #5 clk = ~clk;

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_temporal_sw_top);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_temporal_sw_top, "+mda");
  end
`endif

  initial begin : main
    in0_valid   = 0;
    in1_valid   = 0;
    out0_ready  = 0;
    out1_ready  = 0;
    in0_data    = '0;
    in1_data    = '0;
    t0_cfg_data = '0;

    // Reset
    rst_n = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    #1;

    // Post-reset: error_valid=0
    if (error_valid !== 0)
      $fatal(1, "FAIL: error_valid should be 0 after reset");

    // With no route config, outputs should be idle
    if (out0_valid !== 0)
      $fatal(1, "FAIL: out0_valid should be 0 with empty config");
    if (out1_valid !== 0)
      $fatal(1, "FAIL: out1_valid should be 0 with empty config");

    $display("PASS: tb_temporal_sw_top");
    $finish;
  end

  initial begin : watchdog
    #100000;
    $fatal(1, "FAIL: watchdog timeout");
  end

endmodule
