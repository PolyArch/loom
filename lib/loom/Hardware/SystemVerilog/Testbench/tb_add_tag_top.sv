//===-- tb_add_tag_top.sv - Smoke test for exported add_tag_top -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Minimal smoke testbench for the generated add_tag_top module.
// Validates: reset state, tag attachment, backpressure.
//
//===----------------------------------------------------------------------===//

module tb_add_tag_top;

  logic        clk;
  logic        rst_n;
  logic        in_valid;
  logic        in_ready;
  logic [31:0] in_data;
  logic        out_valid;
  logic        out_ready;
  logic [35:0] out_data;
  logic [3:0]  t0_cfg_data;

  add_tag_top dut (
    .clk        (clk),
    .rst_n      (rst_n),
    .in_valid   (in_valid),
    .in_ready   (in_ready),
    .in_data    (in_data),
    .out_valid  (out_valid),
    .out_ready  (out_ready),
    .out_data   (out_data),
    .t0_cfg_data(t0_cfg_data)
  );

  initial clk = 0;
  always #5 clk = ~clk;

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_add_tag_top);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_add_tag_top, "+mda");
  end
`endif

  initial begin : main
    in_valid    = 0;
    out_ready   = 0;
    in_data     = '0;
    t0_cfg_data = '0;

    // Reset
    rst_n = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    #1;

    // Test: tag attachment
    t0_cfg_data = 4'hA;
    in_data     = 32'hDEAD_BEEF;
    in_valid    = 1;
    out_ready   = 1;
    #1;

    if (out_valid !== 1)
      $fatal(1, "FAIL: out_valid should be 1 when in_valid=1");
    if (out_data !== {4'hA, 32'hDEAD_BEEF})
      $fatal(1, "FAIL: expected {A, DEAD_BEEF}, got %09h", out_data);

    // Test: backpressure
    out_ready = 0;
    #1;
    if (in_ready !== 0)
      $fatal(1, "FAIL: in_ready should be 0 when out_ready=0");

    in_valid = 0;
    @(posedge clk);

    $display("PASS: tb_add_tag_top");
    $finish;
  end

  initial begin : watchdog
    #100000;
    $fatal(1, "FAIL: watchdog timeout");
  end

endmodule
