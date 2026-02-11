//===-- tb_temporal_pe_perport_top.sv - Per-port width temporal PE -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Functional test for temporal PE with per-port widths. The temporal PE uses
// interface tagged(i32, i4) but FU PEs use narrower ports: i16 (adder) and
// i8 (multiplier). The top-level module has 36-bit ports (matching the
// interface type), but the temporal PE instance uses 20-bit per-port wires.
// The top-level adapts between 36-bit and 20-bit via truncation/padding.
//
// Module port layout: {tag[3:0], data[31:0]} = 36 bits
// Per-port (internal): {tag[3:0], data[15:0]} = 20 bits
//
// Instruction encoding (INSN_WIDTH=10, NUM_REGISTERS=0):
//   [0]   valid
//   [4:1] tag (match)
//   [5]   fu_sel (0=adder, 1=multiplier)
//   [9:6] output_tag
//
// Insn 0: tag=0, fu_sel=0 (adder),      output_tag=0  -> 10'h001
// Insn 1: tag=1, fu_sel=1 (multiplier), output_tag=1  -> 10'h063
// cfg_data = {insn1, insn0} = 20'h18C01
//
//===----------------------------------------------------------------------===//

module tb_temporal_pe_perport_top;

  logic        clk;
  logic        rst_n;
  // Module-level ports: 32-bit data + 4-bit tag = 36-bit
  logic        in0_valid, in0_ready;
  logic [35:0] in0_data;
  logic        in1_valid, in1_ready;
  logic [35:0] in1_data;
  logic        out_valid, out_ready;
  logic [35:0] out_data;
  // Config: CONFIG_WIDTH = 2 * INSN_WIDTH = 2 * 10 = 20
  logic [19:0] t0_cfg_data;
  logic        error_valid;
  logic [15:0] error_code;

  temporal_pe_perport_top dut (
    .clk         (clk),
    .rst_n       (rst_n),
    .in0_valid   (in0_valid),
    .in0_ready   (in0_ready),
    .in0_data    (in0_data),
    .in1_valid   (in1_valid),
    .in1_ready   (in1_ready),
    .in1_data    (in1_data),
    .out_valid   (out_valid),
    .out_ready   (out_ready),
    .out_data    (out_data),
    .t0_cfg_data (t0_cfg_data),
    .error_valid (error_valid),
    .error_code  (error_code)
  );

  initial clk = 0;
  always #5 clk = ~clk;

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_temporal_pe_perport_top);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_temporal_pe_perport_top, "+mda");
  end
`endif

  initial begin : main
    integer pass_count;
    integer cycle_count;
    pass_count = 0;
    in0_valid   = 0;
    in1_valid   = 0;
    out_ready   = 0;
    in0_data    = '0;
    in1_data    = '0;
    t0_cfg_data = '0;

    rst_n = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    #1;

    // Check 0: no error after reset
    if (error_valid !== 0) begin : check_reset
      $fatal(1, "error_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    // ---- Configure instruction memory ----
    // Insn 0: valid=1, tag=0, fu_sel=0 (adder), output_tag=0
    //   = 10'b0000_0_0000_1 = 10'h001
    // Insn 1: valid=1, tag=1, fu_sel=1 (multiplier), output_tag=1
    //   = 10'b0001_1_0001_1 = 10'h063
    // cfg_data[9:0] = insn0, cfg_data[19:10] = insn1
    t0_cfg_data = 20'h18C01;

    // ---- Test 1: adder (tag=0) ----
    // Module port: {tag=0, data=0x00001234} = 36'h0_00001234
    // Top-level adapts 32-bit data -> 16-bit for temporal PE: lower 16 bits
    // So FU sees 0x1234 + 0x5678 = 0x68AC
    // Output: temporal PE produces 16-bit {tag=0, 0x68AC}, adapted to 36-bit
    in0_valid = 1;
    in1_valid = 1;
    in0_data  = {4'h0, 32'h0000_1234};
    in1_data  = {4'h0, 32'h0000_5678};
    out_ready = 1;

    // Wait for output (add #1 after posedge for Verilator NBA settling)
    cycle_count = 0;
    while (!out_valid && cycle_count < 20) begin : wait_add
      @(posedge clk); #1;
      cycle_count = cycle_count + 1;
    end

    if (!out_valid) begin : check_add_valid
      $fatal(1, "timeout waiting for adder output");
    end
    // Output adapted from 20-bit to 36-bit: {tag[3:0], 16'b0, data[15:0]}
    // = {4'h0, 16'b0, 16'h68AC} = 36'h0_000068AC
    if (out_data !== 36'h0_0000_68AC) begin : check_add_data
      $fatal(1, "adder: expected 0x0_000068AC, got 0x%09h", out_data);
    end
    pass_count = pass_count + 1;
    $display("[%0t] PASS test 1: adder 0x1234 + 0x5678 = 0x68AC", $time);

    // Deassert inputs, wait for pipeline to drain
    in0_valid = 0;
    in1_valid = 0;
    repeat (3) @(posedge clk);

    // ---- Test 2: multiplier (tag=1) ----
    // Module port: {tag=1, data=0x00000005} and {tag=1, data=0x00000003}
    // Adapted to 16-bit for temporal PE: lower 16 bits = 0x0005 and 0x0003
    // muli uses i8 ports: lower 8 bits = 5 and 3
    // 5 * 3 = 15 = 0x0F -> i8 output -> padded to 16 -> padded to 32
    in0_valid = 1;
    in1_valid = 1;
    in0_data  = {4'h1, 32'h0000_0005};
    in1_data  = {4'h1, 32'h0000_0003};

    cycle_count = 0;
    while (!out_valid && cycle_count < 20) begin : wait_mul
      @(posedge clk); #1;
      cycle_count = cycle_count + 1;
    end

    if (!out_valid) begin : check_mul_valid
      $fatal(1, "timeout waiting for multiplier output");
    end
    // Output: {tag=1, 16'b0, 16'h000F} = 36'h1_0000000F
    if (out_data !== 36'h1_0000_000F) begin : check_mul_data
      $fatal(1, "multiplier: expected 0x1_0000000F, got 0x%09h", out_data);
    end
    pass_count = pass_count + 1;
    $display("[%0t] PASS test 2: multiplier 5 * 3 = 15", $time);

    // Deassert inputs, wait for pipeline to drain
    in0_valid = 0;
    in1_valid = 0;
    repeat (3) @(posedge clk);

    // ---- Test 3: adder with i16 boundary values ----
    // 0x7FFF + 0x0001 = 0x8000
    in0_valid = 1;
    in1_valid = 1;
    in0_data  = {4'h0, 32'h0000_7FFF};
    in1_data  = {4'h0, 32'h0000_0001};

    cycle_count = 0;
    while (!out_valid && cycle_count < 20) begin : wait_add2
      @(posedge clk); #1;
      cycle_count = cycle_count + 1;
    end

    if (!out_valid) begin : check_add2_valid
      $fatal(1, "timeout waiting for adder output (boundary)");
    end
    if (out_data !== 36'h0_0000_8000) begin : check_add2_data
      $fatal(1, "adder boundary: expected 0x0_00008000, got 0x%09h", out_data);
    end
    pass_count = pass_count + 1;
    $display("[%0t] PASS test 3: adder 0x7FFF + 0x0001 = 0x8000", $time);

    // Deassert inputs
    in0_valid = 0;
    in1_valid = 0;
    repeat (3) @(posedge clk);

    // ---- Test 4: multiplier with i8 overflow ----
    // data=0xFF * 0x02 = 0x1FE -> i8 output truncates to 0xFE
    // padded to 16 -> 0x00FE, padded to 32 -> 0x000000FE
    in0_valid = 1;
    in1_valid = 1;
    in0_data  = {4'h1, 32'h0000_00FF};
    in1_data  = {4'h1, 32'h0000_0002};

    cycle_count = 0;
    while (!out_valid && cycle_count < 20) begin : wait_mul2
      @(posedge clk); #1;
      cycle_count = cycle_count + 1;
    end

    if (!out_valid) begin : check_mul2_valid
      $fatal(1, "timeout waiting for multiplier output (overflow)");
    end
    if (out_data !== 36'h1_0000_00FE) begin : check_mul2_data
      $fatal(1, "multiplier overflow: expected 0x1_000000FE, got 0x%09h", out_data);
    end
    pass_count = pass_count + 1;
    $display("[%0t] PASS test 4: multiplier 0xFF * 2 = 0xFE (i8 wrap)", $time);

    in0_valid = 0;
    in1_valid = 0;
    @(posedge clk);

    $display("PASS: tb_temporal_pe_perport_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : watchdog
    #100000;
    $fatal(1, "watchdog timeout");
  end

endmodule
