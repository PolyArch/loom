//===-- tb_fabric_pe_constant.sv - Parameterized constant PE test -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_fabric_pe_constant;
  parameter int DATA_WIDTH = 32;
  parameter int TAG_WIDTH  = 0;

  localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH;
  localparam int SAFE_PW = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1;
  localparam int CONFIG_WIDTH = (TAG_WIDTH > 0) ? DATA_WIDTH + TAG_WIDTH : DATA_WIDTH;

  logic clk, rst_n;
  logic in_valid, in_ready;
  logic [SAFE_PW-1:0] in_data;
  logic out_valid, out_ready;
  logic [SAFE_PW-1:0] out_data;
  logic [CONFIG_WIDTH-1:0] cfg_data;

  fabric_pe_constant #(
    .DATA_WIDTH(DATA_WIDTH),
    .TAG_WIDTH(TAG_WIDTH)
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
    cfg_data = {DATA_WIDTH{1'b1}};

    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // Test: send control token, expect constant output
    in_valid = 1;
    @(posedge clk);
    while (!in_ready) @(posedge clk);

    if (out_valid !== 1'b1) begin : check_valid
      $fatal(1, "out_valid should be 1");
    end
    if (out_data[DATA_WIDTH-1:0] !== {DATA_WIDTH{1'b1}}) begin : check_data
      $fatal(1, "constant value mismatch");
    end
    pass_count = pass_count + 1;

    in_valid = 0;

    $display("PASS: tb_fabric_pe_constant DW=%0d TW=%0d (%0d checks)",
             DATA_WIDTH, TAG_WIDTH, pass_count);
    $finish;
  end

  initial begin : timeout
    #10000;
    $fatal(1, "TIMEOUT");
  end
endmodule
