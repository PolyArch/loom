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
  logic [19:0] t0_cfg_data;
  logic        error_valid;
  logic [15:0] error_code;

  temporal_pe_top dut (.*);

  initial begin : clk_gen
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin : test
    integer pass_count;
    integer cycle_count;
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

    // Instruction memory layout (INSN_WIDTH=10, NUM_REGISTERS=0):
    //   [0]   valid
    //   [4:1] tag
    //   [5]   fu_sel
    //   [9:6] output_tag
    //
    // Insn 0: valid=1, tag=0, fu_sel=0 (adder), output_tag=0  -> 10'h001
    // Insn 1: valid=1, tag=1, fu_sel=1 (muli),  output_tag=1  -> 10'h063
    t0_cfg_data = 20'h18C01;

    // Check 1: adder path (tag=0): 0x1234 + 0x5678 = 0x68AC
    in0_data = {4'h0, 32'h0000_1234};
    in1_data = {4'h0, 32'h0000_5678};
    in0_valid = 1'b1;
    in1_valid = 1'b1;

    cycle_count = 0;
    while (!out_valid && cycle_count < 20) begin : wait_add
      @(posedge clk);
      #1;
      cycle_count = cycle_count + 1;
    end
    if (!out_valid) begin : check_add_timeout
      $fatal(1, "timeout waiting for adder output");
    end
    if (out_data !== 36'h0_0000_68AC) begin : check_add_data
      $fatal(1, "adder output mismatch: expected 0x0_000068AC, got 0x%09h", out_data);
    end
    if (error_valid !== 1'b0) begin : check_add_no_error
      $fatal(1, "unexpected error after adder path: code=%0d", error_code);
    end
    pass_count = pass_count + 1;

    in0_valid = 1'b0;
    in1_valid = 1'b0;
    repeat (3) @(posedge clk);

    // Check 2: multiplier path (tag=1): 5 * 3 = 15
    in0_data = {4'h1, 32'h0000_0005};
    in1_data = {4'h1, 32'h0000_0003};
    in0_valid = 1'b1;
    in1_valid = 1'b1;

    cycle_count = 0;
    while (!out_valid && cycle_count < 30) begin : wait_mul
      @(posedge clk);
      #1;
      cycle_count = cycle_count + 1;
    end
    if (!out_valid) begin : check_mul_timeout
      $fatal(1, "timeout waiting for multiplier output");
    end
    if (out_data !== 36'h1_0000_000F) begin : check_mul_data
      $fatal(1, "multiplier output mismatch: expected 0x1_0000000F, got 0x%09h", out_data);
    end
    if (error_valid !== 1'b0) begin : check_mul_no_error
      $fatal(1, "unexpected error after multiplier path: code=%0d", error_code);
    end
    pass_count = pass_count + 1;

    in0_valid = 1'b0;
    in1_valid = 1'b0;
    @(posedge clk);

    $display("PASS: temporal_pe_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : timeout
    #10000;
    $fatal(1, "TIMEOUT");
  end
endmodule
