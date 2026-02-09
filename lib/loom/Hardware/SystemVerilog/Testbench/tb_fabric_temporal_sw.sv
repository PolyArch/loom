//===-- tb_fabric_temporal_sw.sv - Parameterized temporal switch test -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_fabric_temporal_sw;
  parameter int NUM_INPUTS      = 2;
  parameter int NUM_OUTPUTS     = 2;
  parameter int DATA_WIDTH      = 32;
  parameter int TAG_WIDTH       = 4;
  parameter int NUM_ROUTE_TABLE = 4;

  localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH;
  localparam int SAFE_PW = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1;
  localparam int NUM_CONNECTED = NUM_OUTPUTS * NUM_INPUTS; // full crossbar
  localparam int ENTRY_WIDTH = 1 + TAG_WIDTH + NUM_CONNECTED;
  localparam int CONFIG_WIDTH = NUM_ROUTE_TABLE * ENTRY_WIDTH;

  logic clk, rst_n;
  logic [NUM_INPUTS-1:0]              in_valid;
  logic [NUM_INPUTS-1:0]              in_ready;
  logic [NUM_INPUTS*SAFE_PW-1:0]      in_data;
  logic [NUM_OUTPUTS-1:0]             out_valid;
  logic [NUM_OUTPUTS-1:0]             out_ready;
  logic [NUM_OUTPUTS*SAFE_PW-1:0]     out_data;
  logic [CONFIG_WIDTH-1:0]            cfg_data;
  logic        error_valid;
  logic [15:0] error_code;

  fabric_temporal_sw #(
    .NUM_INPUTS(NUM_INPUTS),
    .NUM_OUTPUTS(NUM_OUTPUTS),
    .DATA_WIDTH(DATA_WIDTH),
    .TAG_WIDTH(TAG_WIDTH),
    .NUM_ROUTE_TABLE(NUM_ROUTE_TABLE)
  ) dut (
    .clk(clk), .rst_n(rst_n),
    .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data),
    .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
    .cfg_data(cfg_data),
    .error_valid(error_valid), .error_code(error_code)
  );

  initial begin : clk_gen
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin : test
    integer pass_count;
    pass_count = 0;
    rst_n = 0;
    in_valid = '0;
    out_ready = '1;
    in_data = '0;
    cfg_data = '0;

    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // Verify reset state: no error
    if (error_valid !== 1'b0) begin : check_reset
      $fatal(1, "error_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    $display("PASS: tb_fabric_temporal_sw NI=%0d NO=%0d DW=%0d TW=%0d (%0d checks)",
             NUM_INPUTS, NUM_OUTPUTS, DATA_WIDTH, TAG_WIDTH, pass_count);
    $finish;
  end

  initial begin : timeout
    #10000;
    $fatal(1, "TIMEOUT");
  end
endmodule
