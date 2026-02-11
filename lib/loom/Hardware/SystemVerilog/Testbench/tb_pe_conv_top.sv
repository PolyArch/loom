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
  logic        out_u_valid, out_u_ready;
  logic [15:0] out_u_data;
  logic        error_valid;
  logic [15:0] error_code;

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
    .out_data  (out_data),
    .out_u_valid (out_u_valid),
    .out_u_ready (out_u_ready),
    .out_u_data  (out_u_data),
    .bcast_in0_cfg_route_table(2'b11),
    .bcast_in1_cfg_route_table(2'b11),
    .error_valid (error_valid),
    .error_code  (error_code)
  );

  initial clk = 0;
  always #5 clk = ~clk;

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_pe_conv_top);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_pe_conv_top, "+mda");
  end
`endif

  initial begin : main
    integer pass_count;
    pass_count = 0;
    in0_valid   = 0;
    in1_valid   = 0;
    out_ready   = 0;
    out_u_ready = 0;
    in0_data    = '0;
    in1_data    = '0;

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
    in0_valid   = 1;
    in0_data    = 16'd3;
    in1_valid   = 1;
    in1_data    = 32'd5;
    out_ready   = 1;
    out_u_ready = 1;

    @(posedge clk);
    #1;

    $display("[%0t] DEBUG pe_conv after posedge:", $time);
    $display("  in0_valid=%0b in0_ready=%0b in1_valid=%0b in1_ready=%0b",
             in0_valid, in0_ready, in1_valid, in1_ready);
    $display("  out_valid=%0b out_ready=%0b out_data=0x%04h",
             out_valid, out_ready, out_data);
    $display("  out_u_valid=%0b out_u_ready=%0b out_u_data=0x%04h",
             out_u_valid, out_u_ready, out_u_data);
    $display("  error_valid=%0b error_code=0x%04h", error_valid, error_code);

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

    // Test 4: extui(0xFFFF) + 1 -> 0x0000FFFF + 1 = 0x00010000 -> trunci = 0
    // extui zero-extends, so 0xFFFF becomes 0x0000FFFF (not 0xFFFFFFFF)
    // Contrast: extsi would give 0xFFFFFFFF + 1 = 0 (same trunci result but
    // different intermediate). For extui: 0x0000FFFF + 1 = 0x00010000 -> trunc = 0.
    // The extsi PE (out) also produces 0 for this input, so we add a case
    // that diverges.
    in0_data = 16'hFFFF;
    in1_data = 32'd1;
    @(posedge clk);
    #1;
    // out (extsi): -1 + 1 = 0 -> 0
    if (out_data !== 16'd0) begin : check_extsi_neg
      $fatal(1, "extsi(-1)+1: expected 0, got %0d", out_data);
    end
    // out_u (extui): 65535 + 1 = 65536 -> trunc = 0
    if (out_u_data !== 16'd0) begin : check_extui_ff
      $fatal(1, "extui(0xFFFF)+1: expected 0, got %0d", out_u_data);
    end
    pass_count = pass_count + 1;

    // Test 5: divergent case: extsi(0x8000) + 0 vs extui(0x8000) + 0
    // extsi(0x8000) = 0xFFFF8000, trunci = 0x8000
    // extui(0x8000) = 0x00008000, trunci = 0x8000
    // Both truncate to same value. Use addition to diverge:
    // extsi(0x8000) + 0x8000 = 0xFFFF8000 + 0x8000 = 0x00000000 -> trunc = 0
    // extui(0x8000) + 0x8000 = 0x00008000 + 0x8000 = 0x00010000 -> trunc = 0
    // Still same... Use a subtraction-like approach via addition:
    // extsi(0x8000) + 0x7FFF = 0xFFFF8000 + 0x7FFF = 0xFFFFFFFF -> trunc = 0xFFFF
    // extui(0x8000) + 0x7FFF = 0x00008000 + 0x7FFF = 0x0000FFFF -> trunc = 0xFFFF
    // Both match. Need a case where upper 16 bits differ after trunci:
    // extsi(0x8000) + 0x00010000 = 0xFFFF8000 + 0x00010000 = 0x00008000 -> 0x8000
    // extui(0x8000) + 0x00010000 = 0x00008000 + 0x00010000 = 0x00018000 -> 0x8000
    // Both truncate identically. The key divergence only shows in upper bits.
    // To observe it: use input that causes upper bits to carry differently.
    // extsi(0x8000) + 0x7FFF0000 = 0xFFFF8000 + 0x7FFF0000 = 0x7FFE8000 -> 0x8000
    // extui(0x8000) + 0x7FFF0000 = 0x00008000 + 0x7FFF0000 = 0x7FFF8000 -> 0x8000
    // The trunci output only sees lower 16 bits. Divergence requires addition
    // that wraps the lower 16 bits differently due to carry from bits 16-31.
    //
    // Actually: extsi(0xFFFE) + 2 = 0xFFFFFFFE + 2 = 0x00000000 -> trunc=0
    //           extui(0xFFFE) + 2 = 0x0000FFFE + 2 = 0x00010000 -> trunc=0
    // Both 0. But:
    // extsi(0xFFFE) + 3 = 0xFFFFFFFE + 3 = 0x00000001 -> trunc=1
    // extui(0xFFFE) + 3 = 0x0000FFFE + 3 = 0x00010001 -> trunc=1
    // Still same trunc. The sign/zero extension only affects bits [31:16],
    // which are truncated away. The lower 16 bits always match.
    // So all trunci outputs match regardless of extsi vs extui for same inputs.
    // The meaningful test is that extui produces correct output at all.
    //
    // Verify: extui(3) + 5 = 8 (matches extsi, sanity check that extui is wired)
    in0_data = 16'd3;
    in1_data = 32'd5;
    @(posedge clk);
    #1;
    if (out_u_data !== 16'd8) begin : check_extui_basic
      $fatal(1, "extui(3)+5: expected 8, got %0d", out_u_data);
    end
    pass_count = pass_count + 1;

    // Test 6: extui(0x8000) + 0 = 0x00008000 + 0 = 0x8000 -> trunc = 0x8000
    // Verifies zero-extension of MSB=1 input (would be negative under extsi)
    in0_data = 16'h8000;
    in1_data = 32'd0;
    @(posedge clk);
    #1;
    if (out_u_data !== 16'h8000) begin : check_extui_msb
      $fatal(1, "extui(0x8000)+0: expected 0x8000, got 0x%04h", out_u_data);
    end
    // Also verify extsi gives the same lower-16 result
    if (out_data !== 16'h8000) begin : check_extsi_msb
      $fatal(1, "extsi(0x8000)+0: expected 0x8000, got 0x%04h", out_data);
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
