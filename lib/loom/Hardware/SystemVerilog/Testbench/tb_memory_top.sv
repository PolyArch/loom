//===-- tb_memory_top.sv - E2E smoke test for memory ----------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_memory_top;
  logic clk, rst_n;
  logic ldaddr_valid, ldaddr_ready;
  logic [63:0] ldaddr_data;
  logic staddr_valid, staddr_ready;
  logic [63:0] staddr_data;
  logic stdata_valid, stdata_ready;
  logic [31:0] stdata_data;
  logic lddata_valid, lddata_ready;
  logic [31:0] lddata_data;
  logic lddone_valid, lddone_ready;
  logic lddone_data;
  logic stdone_valid, stdone_ready;
  logic stdone_data;
  logic        error_valid;
  logic [15:0] error_code;

  memory_top dut (.*);

  initial begin : clk_gen
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin : test
    integer pass_count;
    pass_count = 0;
    rst_n = 0;
    ldaddr_valid = 0;
    staddr_valid = 0;
    stdata_valid = 0;
    lddata_ready = 1;
    lddone_ready = 1;
    stdone_ready = 1;
    ldaddr_data = '0;
    staddr_data = '0;
    stdata_data = '0;

    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // Verify reset state: no error
    if (error_valid !== 1'b0) begin : check_reset
      $fatal(1, "error_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    $display("PASS: memory_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : timeout
    #10000;
    $fatal(1, "TIMEOUT");
  end
endmodule
