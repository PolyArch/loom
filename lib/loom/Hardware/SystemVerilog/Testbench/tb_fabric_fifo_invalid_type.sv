//===-- tb_fabric_fifo_invalid_type.sv - Minimal CPL_ test ----*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Minimal testbench that instantiates fabric_fifo with DATA_WIDTH=0 to
// trigger CPL_FIFO_INVALID_TYPE. Uses a fixed DEPTH=1 to avoid structural
// issues from zero-width payload in the circular buffer.
//
//===----------------------------------------------------------------------===//

module tb_fabric_fifo_invalid_type #(
    parameter int DEPTH      = 1,
    parameter int DATA_WIDTH = 0,
    parameter int TAG_WIDTH  = 0,
    parameter bit BYPASSABLE = 0
);

  logic clk;
  logic rst_n;
  logic in_valid, in_ready;
  logic out_valid, out_ready;
  // Use a 1-bit placeholder to avoid zero-width port issues in structural code
  logic [0:0] in_data;
  logic [0:0] out_data;
  logic [0:0] cfg_data;

  fabric_fifo #(
    .DEPTH      (DEPTH),
    .DATA_WIDTH (DATA_WIDTH),
    .TAG_WIDTH  (TAG_WIDTH),
    .BYPASSABLE (BYPASSABLE)
  ) dut (
    .clk       (clk),
    .rst_n     (rst_n),
    .in_valid  (in_valid),
    .in_ready  (in_ready),
    .in_data   (in_data),
    .out_valid (out_valid),
    .out_ready (out_ready),
    .out_data  (out_data),
    .cfg_data  (cfg_data)
  );

  initial clk = 0;
  always #5 clk = ~clk;

  initial begin : main
    rst_n = 0;
    in_valid = 0;
    out_ready = 0;
    in_data = '0;
    cfg_data = '0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    repeat (5) @(posedge clk);
    // If we reach here, CPL_FIFO_INVALID_TYPE did not fire
    $fatal(1, "FAIL: expected CPL_FIFO_INVALID_TYPE but did not get it");
  end

endmodule
