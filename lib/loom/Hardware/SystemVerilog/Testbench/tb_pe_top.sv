//===-- tb_pe_top.sv - Smoke test for exported pe_top ----------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Minimal smoke testbench for the generated pe_top module.
// Validates: reset state, addition operation, handshake protocol.
//
//===----------------------------------------------------------------------===//

module tb_pe_top;

  logic        clk;
  logic        rst_n;
  logic        in0_valid, in0_ready;
  logic [31:0] in0_data;
  logic        in1_valid, in1_ready;
  logic [31:0] in1_data;
  logic        out_valid, out_ready;
  logic [31:0] out_data;

  pe_top dut (
    .clk       (clk),
    .rst_n     (rst_n),
    .in0_valid (in0_valid),
    .in0_ready (in0_ready),
    .in0_data  (in0_data),
    .in1_valid (in1_valid),
    .in1_ready (in1_ready),
    .in1_data  (in1_data),
    .out_valid (out_valid),
    .out_ready (out_ready),
    .out_data  (out_data)
  );

  initial clk = 0;
  always #5 clk = ~clk;

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_pe_top);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_pe_top, "+mda");
  end
`endif

  initial begin : main
    in0_valid = 0;
    in1_valid = 0;
    out_ready = 0;
    in0_data  = '0;
    in1_data  = '0;

    // Reset
    rst_n = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    #1;

    // Post-reset: out_valid should be 0
    if (out_valid !== 0)
      $fatal(1, "FAIL: out_valid should be 0 after reset");

    // Feed two values: 3 + 5 = 8
    in0_valid = 1;
    in0_data  = 32'd3;
    in1_valid = 1;
    in1_data  = 32'd5;
    out_ready = 1;

    // PE has LATENCY_TYP=1, wait for pipeline to produce
    @(posedge clk);
    #1;

    if (out_valid !== 1)
      $fatal(1, "FAIL: out_valid should be 1 after pipeline delay");
    if (out_data !== 32'd8)
      $fatal(1, "FAIL: expected 8, got %0d", out_data);

    // Second test: 0xFFFF_FFFE + 1 = 0xFFFF_FFFF (overflow wrap)
    in0_data = 32'hFFFF_FFFE;
    in1_data = 32'd1;
    @(posedge clk);
    #1;

    if (out_data !== 32'hFFFF_FFFF)
      $fatal(1, "FAIL: expected FFFFFFFF, got %08h", out_data);

    in0_valid = 0;
    in1_valid = 0;
    @(posedge clk);

    $display("PASS: tb_pe_top");
    $finish;
  end

  initial begin : watchdog
    #100000;
    $fatal(1, "FAIL: watchdog timeout");
  end

endmodule
