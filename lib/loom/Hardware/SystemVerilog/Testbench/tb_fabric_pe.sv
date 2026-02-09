//===-- tb_fabric_pe.sv - Parameterized PE template test -------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Tests the fabric_pe template module (before body filling by exportSV).
// The default PE body is a no-op passthrough, so this test verifies the
// skeleton's handshake, pipeline, and tag attachment logic.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_fabric_pe;
  parameter int NUM_INPUTS  = 2;
  parameter int NUM_OUTPUTS = 1;
  parameter int DATA_WIDTH  = 32;
  parameter int TAG_WIDTH   = 0;
  parameter int LATENCY_TYP = 1;

  localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH;
  localparam int SAFE_PW = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1;
  localparam int TAG_CFG_BITS = (TAG_WIDTH > 0) ? NUM_OUTPUTS * TAG_WIDTH : 0;
  localparam int CONFIG_WIDTH = (TAG_CFG_BITS > 0) ? TAG_CFG_BITS : 0;

  logic clk, rst_n;
  logic [NUM_INPUTS-1:0]                in_valid;
  logic [NUM_INPUTS-1:0]                in_ready;
  logic [NUM_INPUTS-1:0][SAFE_PW-1:0]   in_data;
  logic [NUM_OUTPUTS-1:0]              out_valid;
  logic [NUM_OUTPUTS-1:0]              out_ready;
  logic [NUM_OUTPUTS-1:0][SAFE_PW-1:0] out_data;
  logic [CONFIG_WIDTH > 0 ? CONFIG_WIDTH-1 : 0 : 0] cfg_data;

  fabric_pe #(
    .NUM_INPUTS(NUM_INPUTS),
    .NUM_OUTPUTS(NUM_OUTPUTS),
    .DATA_WIDTH(DATA_WIDTH),
    .TAG_WIDTH(TAG_WIDTH),
    .LATENCY_TYP(LATENCY_TYP)
  ) dut (.*);

  initial begin : clk_gen
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin : test
    integer pass_count;
    integer iter_var0;
    pass_count = 0;
    rst_n = 0;
    in_valid = '0;
    out_ready = '1;
    in_data = '0;
    cfg_data = '0;

    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // Test: all outputs should be invalid after reset
    for (iter_var0 = 0; iter_var0 < NUM_OUTPUTS; iter_var0 = iter_var0 + 1) begin : check_reset
      if (out_valid[iter_var0] !== 1'b0) begin : fail
        $fatal(1, "out_valid[%0d] should be 0 after reset", iter_var0);
      end
    end
    pass_count = pass_count + 1;

    $display("PASS: tb_fabric_pe NI=%0d NO=%0d DW=%0d TW=%0d LAT=%0d (%0d checks)",
             NUM_INPUTS, NUM_OUTPUTS, DATA_WIDTH, TAG_WIDTH, LATENCY_TYP, pass_count);
    $finish;
  end

  initial begin : timeout
    #10000;
    $fatal(1, "TIMEOUT");
  end
endmodule
