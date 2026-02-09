//===-- tb_del_tag_top.sv - E2E smoke test for del_tag --------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_del_tag_top;
  logic clk, rst_n;
  logic in_valid, in_ready;
  logic [35:0] in_data;
  logic out_valid, out_ready;
  logic [31:0] out_data;

  del_tag_top dut (.*);

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

    // Test: send tagged value {tag=4'hA, data=32'hDEAD_BEEF}
    // del_tag is combinational, so output appears in the same cycle as input.
    in_valid = 1;
    in_data = {4'hA, 32'hDEAD_BEEF};
    @(posedge clk);
    while (!in_ready) @(posedge clk);
    // Capture output while handshake is active (combinational path)
    if (!out_valid) begin : check_valid
      $fatal(1, "out_valid should be high during handshake");
    end
    if (out_data !== 32'hDEAD_BEEF) begin : check_data
      $display("FAIL: expected DEAD_BEEF, got %h", out_data);
      $fatal(1, "del_tag data mismatch");
    end
    in_valid = 0;
    pass_count = pass_count + 1;

    $display("PASS: del_tag_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : timeout
    #10000;
    $fatal(1, "TIMEOUT");
  end
endmodule
