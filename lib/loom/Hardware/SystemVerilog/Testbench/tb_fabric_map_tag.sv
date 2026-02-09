//===-- tb_fabric_map_tag.sv - Parameterized map_tag test ------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_fabric_map_tag;
  parameter int DATA_WIDTH    = 32;
  parameter int IN_TAG_WIDTH  = 4;
  parameter int OUT_TAG_WIDTH = 4;
  parameter int TABLE_SIZE    = 4;

  localparam int ENTRY_WIDTH  = 1 + IN_TAG_WIDTH + OUT_TAG_WIDTH;
  localparam int IN_PW        = DATA_WIDTH + IN_TAG_WIDTH;
  localparam int OUT_PW       = DATA_WIDTH + OUT_TAG_WIDTH;
  localparam int CONFIG_WIDTH = TABLE_SIZE * ENTRY_WIDTH;

  logic clk, rst_n;
  logic in_valid, in_ready;
  logic [IN_PW-1:0] in_data;
  logic out_valid, out_ready;
  logic [OUT_PW-1:0] out_data;
  logic [CONFIG_WIDTH > 0 ? CONFIG_WIDTH-1 : 0 : 0] cfg_data;
  logic        error_valid;
  logic [15:0] error_code;

  fabric_map_tag #(
    .DATA_WIDTH(DATA_WIDTH),
    .IN_TAG_WIDTH(IN_TAG_WIDTH),
    .OUT_TAG_WIDTH(OUT_TAG_WIDTH),
    .TABLE_SIZE(TABLE_SIZE)
  ) dut (.*);

  initial begin : clk_gen
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin : test
    integer pass_count;
    pass_count = 0;
    rst_n = 0;
    in_valid = 0;
    out_ready = 1;
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

    $display("PASS: tb_fabric_map_tag DW=%0d ITW=%0d OTW=%0d TS=%0d (%0d checks)",
             DATA_WIDTH, IN_TAG_WIDTH, OUT_TAG_WIDTH, TABLE_SIZE, pass_count);
    $finish;
  end

  initial begin : timeout
    #10000;
    $fatal(1, "TIMEOUT");
  end
endmodule
