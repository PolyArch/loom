//===-- tb_map_tag_top.sv - E2E smoke test for map_tag --------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_map_tag_top;
  logic clk, rst_n;
  logic in_valid, in_ready;
  logic [35:0] in_data;
  logic out_valid, out_ready;
  logic [35:0] out_data;
  logic [35:0] m0_cfg_data;
  logic        error_valid;
  logic [15:0] error_code;

  map_tag_top dut (.*);

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
    m0_cfg_data = '0;

    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // Verify reset state: no error
    if (error_valid !== 1'b0) begin : check_reset
      $fatal(1, "error_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    $display("PASS: map_tag_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : timeout
    #10000;
    $fatal(1, "TIMEOUT");
  end
endmodule
