//===-- tb_pe_constant_top.sv - E2E smoke test for pe_constant -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_pe_constant_top;
  logic clk, rst_n;
  logic ctrl_valid, ctrl_ready;
  logic ctrl_data;
  logic out_valid, out_ready;
  logic [31:0] out_data;
  logic [31:0] c0_cfg_data;

  pe_constant_top dut (.*);

  initial begin : clk_gen
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin : test
    integer pass_count;
    pass_count = 0;
    rst_n = 0;
    ctrl_valid = 0;
    out_ready = 1;
    ctrl_data = 0;
    c0_cfg_data = 32'h0000_002A; // constant = 42

    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // Send control token to trigger constant output
    ctrl_valid = 1;
    ctrl_data = 1'b1;
    @(posedge clk);
    while (!ctrl_ready) @(posedge clk);
    ctrl_valid = 0;

    // Wait for output
    while (!out_valid) @(posedge clk);
    if (out_data !== 32'h0000_002A) begin : check_data
      $display("FAIL: expected 0000002A, got %h", out_data);
      $fatal(1, "pe_constant data mismatch");
    end
    pass_count = pass_count + 1;

    $display("PASS: pe_constant_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : timeout
    #10000;
    $fatal(1, "TIMEOUT");
  end
endmodule
