//===-- tb_fifo_top.sv - Smoke test for exported fifo_top ------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Minimal smoke testbench for the generated fifo_top module.
// Validates: reset state, single push/pop, handshake protocol.
//
//===----------------------------------------------------------------------===//

module tb_fifo_top;

  logic        clk;
  logic        rst_n;
  logic        in_valid;
  logic        in_ready;
  logic [31:0] in_data;
  logic        out_valid;
  logic        out_ready;
  logic [31:0] out_data;

  fifo_top dut (
    .clk       (clk),
    .rst_n     (rst_n),
    .in_valid  (in_valid),
    .in_ready  (in_ready),
    .in_data   (in_data),
    .out_valid (out_valid),
    .out_ready (out_ready),
    .out_data  (out_data)
  );

  initial clk = 0;
  always #5 clk = ~clk;

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_fifo_top);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_fifo_top, "+mda");
  end
`endif

  initial begin : main
    in_valid  = 0;
    out_ready = 0;
    in_data   = '0;

    // Reset
    rst_n = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    #1;

    // Post-reset: out_valid=0, in_ready=1
    if (out_valid !== 0)
      $fatal(1, "FAIL: out_valid should be 0 after reset");
    if (in_ready !== 1)
      $fatal(1, "FAIL: in_ready should be 1 after reset");

    // Push one value
    in_valid = 1;
    in_data  = 32'hCAFE;
    @(posedge clk);
    #1;
    in_valid = 0;

    // Check output appears
    if (out_valid !== 1)
      $fatal(1, "FAIL: out_valid should be 1 after push");
    if (out_data !== 32'hCAFE)
      $fatal(1, "FAIL: data mismatch: expected CAFE got %08h", out_data);

    // Pop
    out_ready = 1;
    @(posedge clk);
    #1;
    out_ready = 0;

    if (out_valid !== 0)
      $fatal(1, "FAIL: out_valid should be 0 after pop");

    $display("PASS: tb_fifo_top");
    $finish;
  end

  initial begin : watchdog
    #100000;
    $fatal(1, "FAIL: watchdog timeout");
  end

endmodule
