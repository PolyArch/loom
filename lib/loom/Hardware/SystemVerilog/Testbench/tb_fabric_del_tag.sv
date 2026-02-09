//===-- tb_fabric_del_tag.sv - Parameterized del_tag test ------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_fabric_del_tag;
  parameter int DATA_WIDTH = 32;
  parameter int TAG_WIDTH  = 4;

  localparam int IN_PW  = DATA_WIDTH + TAG_WIDTH;
  localparam int OUT_PW = DATA_WIDTH;

  logic clk, rst_n;
  logic in_valid, in_ready;
  logic [IN_PW-1:0] in_data;
  logic out_valid, out_ready;
  logic [OUT_PW-1:0] out_data;

  fabric_del_tag #(
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

    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // Test 1: strip tag, pass value through
    in_valid = 1;
    in_data = {{TAG_WIDTH{1'b1}}, {DATA_WIDTH{1'b1}}};
    @(posedge clk);
    while (!in_ready) @(posedge clk);

    if (out_valid !== 1'b1) begin : check_valid
      $fatal(1, "out_valid should be 1");
    end
    if (out_data !== {DATA_WIDTH{1'b1}}) begin : check_data
      $fatal(1, "out_data mismatch");
    end
    pass_count = pass_count + 1;

    // Test 2: backpressure
    out_ready = 0;
    in_valid = 1;
    in_data = {{TAG_WIDTH{1'b0}}, {DATA_WIDTH{1'b0}}};
    @(posedge clk);
    if (in_ready !== 1'b0) begin : check_bp
      $fatal(1, "in_ready should be 0 under backpressure");
    end
    pass_count = pass_count + 1;

    in_valid = 0;
    out_ready = 1;

    $display("PASS: tb_fabric_del_tag DW=%0d TW=%0d (%0d checks)",
             DATA_WIDTH, TAG_WIDTH, pass_count);
    $finish;
  end

  initial begin : timeout
    #10000;
    $fatal(1, "TIMEOUT");
  end
endmodule
