//===-- tb_temporal_pe_compare_top.sv - Temporal PE compare FU test -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Functional test for temporal PE containing a compare FU (arith.cmpi) and an
// arithmetic FU (arith.addi).  The temporal PE uses interface tagged(i16, i2)
// with 2 FUs, 2 instructions, and 0 registers.
//
// Module port layout: {tag[1:0], data[15:0]} = 18 bits
//
// Config layout (CONFIG_WIDTH = 16):
//   [3:0]   FU0 cmpi predicate (4 bits, FU_CMP_CFG_BITS)
//   [9:4]   insn0 (INSN_WIDTH = 6)
//   [15:10] insn1 (INSN_WIDTH = 6)
//
// INSN_WIDTH = 6:
//   [0]   valid
//   [2:1] tag (match)
//   [3]   fu_sel (0=cmpi, 1=addi)
//   [5:4] output_tag
//
// Insn 0: tag=0, fu_sel=0 (cmpi), output_tag=0 -> 6'b00_0_00_1 = 6'h01
// Insn 1: tag=1, fu_sel=1 (addi), output_tag=1 -> 6'b01_1_01_1 = 6'h1B
//
// Test configs:
//   pred=0 (eq):  {6'h1B, 6'h01, 4'h0} = 16'h6C10
//   pred=2 (slt): {6'h1B, 6'h01, 4'h2} = 16'h6C12
//   pred=10 (invalid): {6'h1B, 6'h01, 4'hA} = 16'h6C1A
//
//===----------------------------------------------------------------------===//

`include "fabric_error.svh"

module tb_temporal_pe_compare_top;

  logic        clk;
  logic        rst_n;
  // Module-level ports: 16-bit data + 2-bit tag = 18-bit
  logic        in0_valid, in0_ready;
  logic [17:0] in0_data;
  logic        in1_valid, in1_ready;
  logic [17:0] in1_data;
  logic        out_valid, out_ready;
  logic [17:0] out_data;
  // Config: CONFIG_WIDTH = 16
  logic [15:0] t0_cfg_data;
  logic        error_valid;
  logic [15:0] error_code;

  temporal_pe_compare_top dut (
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
    $dumpvars(0, tb_temporal_pe_compare_top);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_temporal_pe_compare_top, "+mda");
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

    // ---- Configure: pred=0 (eq) ----
    // cfg_data = {insn1=6'h1B, insn0=6'h01, pred=4'h0} = 16'h6C10
    t0_cfg_data = 16'h6C10;

    // ---- Test 1: cmpi eq (tag=0) ----
    // a=100, b=100 -> eq is true (1)
    // Input: {tag=2'b00, data=16'd100} = {2'b00, 16'h0064}
    // Output: {tag=2'b00, 16'h0001} (i1 result zero-extended to 16)
    in0_valid = 1;
    in1_valid = 1;
    in0_data  = {2'b00, 16'h0064};
    in1_data  = {2'b00, 16'h0064};
    out_ready = 1;

    cycle_count = 0;
    while (!out_valid && cycle_count < 20) begin : wait_eq
      @(posedge clk); #1;
      cycle_count = cycle_count + 1;
    end

    if (!out_valid) begin : check_eq_valid
      $fatal(1, "timeout waiting for cmpi eq output");
    end
    if (out_data !== {2'b00, 16'h0001}) begin : check_eq_data
      $fatal(1, "cmpi eq: expected 0x%05h, got 0x%05h",
             {2'b00, 16'h0001}, out_data);
    end
    pass_count = pass_count + 1;
    $display("[%0t] PASS test 1: cmpi eq 100==100 -> 1", $time);

    // Deassert inputs, drain
    in0_valid = 0;
    in1_valid = 0;
    repeat (3) @(posedge clk);

    // ---- Test 2: runtime switch to slt (tag=0) ----
    // Change pred to 2 (slt): cfg_data = 16'h6C12
    // a=50, b=100 -> 50 < 100 is true (1)
    t0_cfg_data = 16'h6C12;
    in0_valid = 1;
    in1_valid = 1;
    in0_data  = {2'b00, 16'h0032}; // 50
    in1_data  = {2'b00, 16'h0064}; // 100
    out_ready = 1;

    cycle_count = 0;
    while (!out_valid && cycle_count < 20) begin : wait_slt
      @(posedge clk); #1;
      cycle_count = cycle_count + 1;
    end

    if (!out_valid) begin : check_slt_valid
      $fatal(1, "timeout waiting for cmpi slt output");
    end
    if (out_data !== {2'b00, 16'h0001}) begin : check_slt_data
      $fatal(1, "cmpi slt: expected 0x%05h, got 0x%05h",
             {2'b00, 16'h0001}, out_data);
    end
    pass_count = pass_count + 1;
    $display("[%0t] PASS test 2: cmpi slt 50<100 -> 1", $time);

    in0_valid = 0;
    in1_valid = 0;
    repeat (3) @(posedge clk);

    // ---- Test 3: addi (tag=1) ----
    // a=0x1234, b=0x5678 -> 0x68AC
    // Input: {tag=2'b01, data=16'h1234} etc.
    // Output: {tag=2'b01, 16'h68AC}
    in0_valid = 1;
    in1_valid = 1;
    in0_data  = {2'b01, 16'h1234};
    in1_data  = {2'b01, 16'h5678};
    out_ready = 1;

    cycle_count = 0;
    while (!out_valid && cycle_count < 20) begin : wait_add
      @(posedge clk); #1;
      cycle_count = cycle_count + 1;
    end

    if (!out_valid) begin : check_add_valid
      $fatal(1, "timeout waiting for addi output");
    end
    if (out_data !== {2'b01, 16'h68AC}) begin : check_add_data
      $fatal(1, "addi: expected 0x%05h, got 0x%05h",
             {2'b01, 16'h68AC}, out_data);
    end
    pass_count = pass_count + 1;
    $display("[%0t] PASS test 3: addi 0x1234+0x5678 = 0x68AC", $time);

    in0_valid = 0;
    in1_valid = 0;
    repeat (3) @(posedge clk);

    // ---- Test 4: invalid predicate error ----
    // Set pred=10 (invalid): cfg_data = 16'h6C1A
    // Error propagates through two latches: temporal PE error_latch then
    // top-level error_latch, so we need 2 clock edges.
    t0_cfg_data = 16'h6C1A;
    @(posedge clk); // temporal PE error latch captures
    @(posedge clk); // top module error latch captures
    #1;

    if (!error_valid) begin : check_err_assert
      $fatal(1, "error_valid not asserted for cmpi predicate=10");
    end
    if (error_code !== CFG_PE_CMPI_PREDICATE_INVALID) begin : check_err_code
      $fatal(1, "wrong error_code: expected %0d got %0d",
             CFG_PE_CMPI_PREDICATE_INVALID, error_code);
    end
    pass_count = pass_count + 1;
    $display("[%0t] PASS test 4a: invalid predicate detected", $time);

    // Reset clears the error latch
    rst_n = 0;
    t0_cfg_data = 16'h6C10; // restore valid config
    repeat (2) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    #1;

    if (error_valid) begin : check_err_clear
      $fatal(1, "error_valid still asserted after reset with valid predicate");
    end
    pass_count = pass_count + 1;
    $display("[%0t] PASS test 4b: error cleared after reset", $time);

    in0_valid = 0;
    in1_valid = 0;
    @(posedge clk);

    $display("PASS: tb_temporal_pe_compare_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : watchdog
    #100000;
    $fatal(1, "watchdog timeout");
  end

endmodule
