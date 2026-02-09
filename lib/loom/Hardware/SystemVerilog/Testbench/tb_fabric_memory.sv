//===-- tb_fabric_memory.sv - Parameterized memory test --------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_fabric_memory;
  parameter int DATA_WIDTH  = 32;
  parameter int TAG_WIDTH   = 0;
  parameter int LD_COUNT    = 1;
  parameter int ST_COUNT    = 1;
  parameter int LSQ_DEPTH   = 4;
  parameter int IS_PRIVATE  = 1;
  parameter int MEM_DEPTH   = 64;

  localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH;
  localparam int SAFE_PW = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1;
  localparam int NUM_INPUTS  = LD_COUNT + 2 * ST_COUNT;
  localparam int NUM_OUTPUTS = (IS_PRIVATE ? 0 : 1) + LD_COUNT + 1 + (ST_COUNT > 0 ? 1 : 0);

  logic clk, rst_n;
  logic [NUM_INPUTS-1:0]                in_valid;
  logic [NUM_INPUTS-1:0]                in_ready;
  logic [NUM_INPUTS-1:0][SAFE_PW-1:0]  in_data;
  logic [NUM_OUTPUTS-1:0]              out_valid;
  logic [NUM_OUTPUTS-1:0]              out_ready;
  logic [NUM_OUTPUTS-1:0][SAFE_PW-1:0] out_data;
  logic        error_valid;
  logic [15:0] error_code;

  fabric_memory #(
    .DATA_WIDTH(DATA_WIDTH),
    .TAG_WIDTH(TAG_WIDTH),
    .LD_COUNT(LD_COUNT),
    .ST_COUNT(ST_COUNT),
    .LSQ_DEPTH(LSQ_DEPTH),
    .IS_PRIVATE(IS_PRIVATE),
    .MEM_DEPTH(MEM_DEPTH)
  ) dut (
    .clk(clk), .rst_n(rst_n),
    .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data),
    .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
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

    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // Verify reset state: no error
    if (error_valid !== 1'b0) begin : check_reset
      $fatal(1, "error_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    $display("PASS: tb_fabric_memory DW=%0d TW=%0d LD=%0d ST=%0d (%0d checks)",
             DATA_WIDTH, TAG_WIDTH, LD_COUNT, ST_COUNT, pass_count);
    $finish;
  end

  initial begin : timeout
    #10000;
    $fatal(1, "TIMEOUT");
  end
endmodule
