//===-- tb_pe_conv_top.sv - Mixed-width conversion PE test ------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Smoke test for the generated pe_conv_top module. Validates mixed-width
// conversion pipeline: (i16, i32) -> extsi -> addi -> trunci -> i16.
//
//===----------------------------------------------------------------------===//

module tb_pe_conv_top;

  logic        clk;
  logic        rst_n;
  logic        in0_valid, in0_ready;
  logic [15:0] in0_data;
  logic        in1_valid, in1_ready;
  logic [31:0] in1_data;
  logic        out_valid, out_ready;
  logic [15:0] out_data;

  pe_conv_top dut (
    .clk       (clk),
    .rst_n     (rst_n),
    .in0_valid (in0_valid),
    .in0_ready (in0_ready),
    .in0_data  (in0_data),
    .in1_valid (in1_valid),
    .in1_ready (in1_ready),
    .in1_data  (in1_data),
    .out_valid (out_valid),
    .out_ready (out_ready),
    .out_data  (out_data)
  );

  initial clk = 0;
  always #5 clk = ~clk;

  initial begin : main
    integer pass_count;
    pass_count = 0;
    in0_valid = 0;
    in1_valid = 0;
    out_ready = 0;
    in0_data  = '0;
    in1_data  = '0;

    rst_n = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    #1;

    if (out_valid !== 0) begin : check_reset
      $fatal(1, "out_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    // Test 1: extsi(3) + 5 = 8 -> trunci to i16 = 8
    in0_valid = 1;
    in0_data  = 16'd3;
    in1_valid = 1;
    in1_data  = 32'd5;
    out_ready = 1;

    @(posedge clk);
    #1;

    if (out_valid !== 1) begin : check_valid1
      $fatal(1, "out_valid should be 1 after pipeline delay");
    end
    if (out_data !== 16'd8) begin : check_data1
      $fatal(1, "expected 8, got %0d", out_data);
    end
    pass_count = pass_count + 1;

    // Test 2: extsi(-1 as i16 = 0xFFFF) + 1 = 0 -> trunci = 0
    // -1 sign-extends to 0xFFFFFFFF, add 1 = 0x00000000, trunc = 0
    in0_data = 16'hFFFF;
    in1_data = 32'd1;
    @(posedge clk);
    #1;

    if (out_data !== 16'd0) begin : check_data2
      $fatal(1, "expected 0 (sign-extend -1 + 1), got %0d", out_data);
    end
    pass_count = pass_count + 1;

    // Test 3: extsi(0x7FFF) + 0x00010000 = 0x00017FFF -> trunci = 0x7FFF
    // Verifies upper bits are properly truncated
    in0_data = 16'h7FFF;
    in1_data = 32'h0001_0000;
    @(posedge clk);
    #1;

    if (out_data !== 16'h7FFF) begin : check_data3
      $fatal(1, "expected 0x7FFF (truncation), got 0x%04h", out_data);
    end
    pass_count = pass_count + 1;

    in0_valid = 0;
    in1_valid = 0;
    @(posedge clk);

    $display("PASS: tb_pe_conv_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : watchdog
    #100000;
    $fatal(1, "watchdog timeout");
  end

endmodule
