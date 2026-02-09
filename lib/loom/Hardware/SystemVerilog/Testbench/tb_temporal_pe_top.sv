//===-- tb_temporal_pe_top.sv - E2E smoke test for temporal_pe -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_temporal_pe_top;
  logic clk, rst_n;
  logic in0_valid, in0_ready;
  logic [35:0] in0_data;
  logic in1_valid, in1_ready;
  logic [35:0] in1_data;
  logic out_valid, out_ready;
  logic [35:0] out_data;
  logic [17:0] t0_cfg_data;
  logic        error_valid;
  logic [15:0] error_code;

  temporal_pe_top dut (.*);

  initial begin : clk_gen
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin : test
    integer pass_count;
    pass_count = 0;
    rst_n = 0;
    in0_valid = 0;
    in1_valid = 0;
    out_ready = 1;
    in0_data = '0;
    in1_data = '0;
    t0_cfg_data = '0;

    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // Verify reset state: no error
    if (error_valid !== 1'b0) begin : check_reset
      $fatal(1, "error_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    $display("PASS: temporal_pe_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : timeout
    #10000;
    $fatal(1, "TIMEOUT");
  end
endmodule
