//===-- tb_temporal_pe_reg_top.sv - Temporal PE register E2E test -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// End-to-end test for temporal PE register data flow through the generated
// top module. Exercises register write, register read, FIFO ordering, and
// dual-register reads using arith.addi with tagged(i32, i4) interface.
//
// Derived parameters (NUM_REGISTERS=2, NUM_INSTRUCTIONS=4, 1 FU):
//   REG_BITS = 2, FU_SEL_BITS = 0, RES_BITS = 2
//   RESULT_WIDTH = 6, INSN_WIDTH = 15, CONFIG_WIDTH = 60
//
// Instruction bit layout (per 15-bit slot, LSB to MSB):
//   [0]     valid        (1b)  instruction valid
//   [4:1]   insn_tag     (4b)  match tag
//   [5]     op0_reg_idx  (1b)  operand 0 register index
//   [6]     op0_is_reg   (1b)  1=register source, 0=input port
//   [7]     op1_reg_idx  (1b)  operand 1 register index
//   [8]     op1_is_reg   (1b)  1=register source, 0=input port
//   [12:9]  res_tag      (4b)  output tag (must be 0 for reg writes)
//   [13]    res_reg_idx  (1b)  register index for result
//   [14]    res_is_reg   (1b)  1=register write, 0=external output
//
// Module port layout: {tag[3:0], data[31:0]} = 36 bits
//
//===----------------------------------------------------------------------===//

module tb_temporal_pe_reg_top;

  localparam int TAG_WIDTH = 4;
  localparam int REG_BITS = 2;
  localparam int RES_BITS = 2;
  localparam int RESULT_WIDTH = 6;
  localparam int INSN_WIDTH = 15;
  localparam int INSN_VALID_LSB = 0;
  localparam int INSN_TAG_LSB = INSN_VALID_LSB + 1;
  localparam int INSN_OPERANDS_LSB = INSN_TAG_LSB + TAG_WIDTH;
  localparam int OP0_BASE = INSN_OPERANDS_LSB + 0 * REG_BITS;
  localparam int OP1_BASE = INSN_OPERANDS_LSB + 1 * REG_BITS;
  localparam int INSN_RESULTS_LSB = INSN_OPERANDS_LSB + 2 * REG_BITS;
  localparam int RES0_BASE = INSN_RESULTS_LSB + 0 * RESULT_WIDTH;

  logic        clk;
  logic        rst_n;
  // Module-level ports: 32-bit data + 4-bit tag = 36-bit
  logic        in0_valid, in0_ready;
  logic [35:0] in0_data;
  logic        in1_valid, in1_ready;
  logic [35:0] in1_data;
  logic        out_valid, out_ready;
  logic [35:0] out_data;
  // Config: CONFIG_WIDTH = 4 * 15 = 60
  logic [59:0] t0_cfg_data;
  logic        error_valid;
  logic [15:0] error_code;

  temporal_pe_reg_top dut (
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
    $dumpvars(0, tb_temporal_pe_reg_top);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_temporal_pe_reg_top, "+mda");
  end
`endif

  // Helper: build one 15-bit instruction word
  // Arguments: valid, insn_tag, op1_is_reg, op1_reg_idx, op0_is_reg, op0_reg_idx,
  //            res_is_reg, res_reg_idx, res_tag
  function automatic [14:0] make_insn(
    input logic        f_valid,
    input logic [3:0]  f_insn_tag,
    input logic        f_op1_is_reg,
    input logic        f_op1_reg_idx,
    input logic        f_op0_is_reg,
    input logic        f_op0_reg_idx,
    input logic        f_res_is_reg,
    input logic        f_res_reg_idx,
    input logic [3:0]  f_res_tag
  );
  begin : make_insn_body
    make_insn = '0;
    make_insn[INSN_VALID_LSB] = f_valid;
    make_insn[INSN_TAG_LSB +: TAG_WIDTH] = f_insn_tag;
    make_insn[OP0_BASE +: (REG_BITS - 1)] = f_op0_reg_idx;
    make_insn[OP0_BASE + REG_BITS - 1] = f_op0_is_reg;
    make_insn[OP1_BASE +: (REG_BITS - 1)] = f_op1_reg_idx;
    make_insn[OP1_BASE + REG_BITS - 1] = f_op1_is_reg;
    make_insn[RES0_BASE +: TAG_WIDTH] = f_res_tag;
    make_insn[RES0_BASE + TAG_WIDTH +: (RES_BITS - 1)] = f_res_reg_idx;
    make_insn[RES0_BASE + RESULT_WIDTH - 1] = f_res_is_reg;
  end
  endfunction

  initial begin : main
    integer pass_count;
    integer cycle_count;
    logic [14:0] insn0, insn1, insn2, insn3;
    logic in0_seen;
    logic in1_seen;
    logic t3_seen;
    logic [35:0] t3_data;
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

    // ================================================================
    // Config A: basic register write/read and FIFO ordering
    //   Insn 0 (tag=1): op0=in, op1=in -> reg(0)   [write register]
    //   Insn 1 (tag=2): op0=reg(0), op1=in -> out(tag=2) [read register]
    //   Insn 2-3: invalid
    // ================================================================
    insn0 = make_insn(1, 4'd1, 0, 0, 0, 0, 1, 0, 4'd0);
    insn1 = make_insn(1, 4'd2, 0, 0, 1, 0, 0, 0, 4'd2);
    insn2 = 15'd0;
    insn3 = 15'd0;
    t0_cfg_data = {insn3, insn2, insn1, insn0};
    @(posedge clk); #1;

    // ---- Test 1: Basic register write then read ----
    // Fire insn 0 (tag=1): 10 + 20 = 30 -> reg(0)
    in0_valid = 1;
    in1_valid = 1;
    in0_data  = {4'h1, 32'h0000_000A};  // tag=1, data=10
    in1_data  = {4'h1, 32'h0000_0014};  // tag=1, data=20
    out_ready = 1;

    // Wait for inputs to be accepted
    in0_seen = 1'b0;
    in1_seen = 1'b0;
    cycle_count = 0;
    while (!(in0_seen && in1_seen) && cycle_count < 20) begin : wait_wr1_accept
      @(posedge clk); #1;
      if (in0_valid && in0_ready) in0_seen = 1'b1;
      if (in1_valid && in1_ready) in1_seen = 1'b1;
      cycle_count = cycle_count + 1;
    end
    if (!(in0_seen && in1_seen)) begin : check_wr1_accept
      $fatal(1, "Test 1: timeout waiting write inputs accepted");
    end
    in0_valid = 0;
    in1_valid = 0;
    @(posedge clk); #1;

    // Wait for register write to complete
    repeat (4) @(posedge clk); #1;

    // Fire insn 1 (tag=2): reg(0)=30 + 5 = 35 -> out
    in0_valid = 1;
    in1_valid = 1;
    in0_data  = {4'h2, 32'h0000_0000};  // tag=2, data=don't care (reg sourced)
    in1_data  = {4'h2, 32'h0000_0005};  // tag=2, data=5
    out_ready = 1;

    cycle_count = 0;
    while (!out_valid && cycle_count < 30) begin : wait_rd1_out
      @(posedge clk); #1;
      cycle_count = cycle_count + 1;
    end

    if (!out_valid) begin : check_t1_valid
      $fatal(1, "Test 1: timeout waiting for output");
    end
    // Expected: 30 + 5 = 35 = 0x23, tag=2
    if (out_data !== {4'h2, 32'h0000_0023}) begin : check_t1_data
      $fatal(1, "Test 1: expected {tag=2, 0x23}, got 0x%09h", out_data);
    end
    pass_count = pass_count + 1;
    $display("[%0t] PASS test 1: reg write(30) then read: 30+5=35", $time);

    in0_valid = 0;
    in1_valid = 0;
    repeat (4) @(posedge clk); #1;

    // ---- Test 2: FIFO ordering ----
    // Write reg(0) twice: first 30, then 70
    // Fire insn 0 (tag=1): 10 + 20 = 30 -> reg(0)
    in0_valid = 1;
    in1_valid = 1;
    in0_data  = {4'h1, 32'h0000_000A};  // 10
    in1_data  = {4'h1, 32'h0000_0014};  // 20
    out_ready = 1;

    in0_seen = 1'b0;
    in1_seen = 1'b0;
    cycle_count = 0;
    while (!(in0_seen && in1_seen) && cycle_count < 20) begin : wait_wr2a_accept
      @(posedge clk); #1;
      if (in0_valid && in0_ready) in0_seen = 1'b1;
      if (in1_valid && in1_ready) in1_seen = 1'b1;
      cycle_count = cycle_count + 1;
    end
    if (!(in0_seen && in1_seen)) begin : check_wr2a_accept
      $fatal(1, "Test 2a: timeout waiting first FIFO write accepted");
    end
    in0_valid = 0;
    in1_valid = 0;
    @(posedge clk); #1;
    repeat (4) @(posedge clk); #1;

    // Fire insn 0 (tag=1): 30 + 40 = 70 -> reg(0)
    in0_valid = 1;
    in1_valid = 1;
    in0_data  = {4'h1, 32'h0000_001E};  // 30
    in1_data  = {4'h1, 32'h0000_0028};  // 40
    out_ready = 1;

    in0_seen = 1'b0;
    in1_seen = 1'b0;
    cycle_count = 0;
    while (!(in0_seen && in1_seen) && cycle_count < 20) begin : wait_wr2b_accept
      @(posedge clk); #1;
      if (in0_valid && in0_ready) in0_seen = 1'b1;
      if (in1_valid && in1_ready) in1_seen = 1'b1;
      cycle_count = cycle_count + 1;
    end
    if (!(in0_seen && in1_seen)) begin : check_wr2b_accept
      $fatal(1, "Test 2b: timeout waiting second FIFO write accepted");
    end
    in0_valid = 0;
    in1_valid = 0;
    @(posedge clk); #1;
    repeat (4) @(posedge clk); #1;

    // Read first: reg(0)=30 + 1 = 31 -> out
    in0_valid = 1;
    in1_valid = 1;
    in0_data  = {4'h2, 32'h0000_0000};  // tag=2, reg sourced
    in1_data  = {4'h2, 32'h0000_0001};  // tag=2, data=1
    out_ready = 1;

    cycle_count = 0;
    while (!out_valid && cycle_count < 30) begin : wait_fifo1_out
      @(posedge clk); #1;
      cycle_count = cycle_count + 1;
    end

    if (!out_valid) begin : check_t2a_valid
      $fatal(1, "Test 2a: timeout waiting for FIFO head output");
    end
    // Expected: 30 + 1 = 31 = 0x1F, tag=2
    if (out_data !== {4'h2, 32'h0000_001F}) begin : check_t2a_data
      $fatal(1, "Test 2a: expected {tag=2, 0x1F} (FIFO head=30), got 0x%09h", out_data);
    end
    pass_count = pass_count + 1;
    $display("[%0t] PASS test 2a: FIFO head 30+1=31", $time);

    in0_valid = 0;
    in1_valid = 0;
    repeat (4) @(posedge clk); #1;

    // Read second: reg(0)=70 + 1 = 71 -> out
    in0_valid = 1;
    in1_valid = 1;
    in0_data  = {4'h2, 32'h0000_0000};
    in1_data  = {4'h2, 32'h0000_0001};
    out_ready = 1;

    cycle_count = 0;
    while (!out_valid && cycle_count < 30) begin : wait_fifo2_out
      @(posedge clk); #1;
      cycle_count = cycle_count + 1;
    end

    if (!out_valid) begin : check_t2b_valid
      $fatal(1, "Test 2b: timeout waiting for FIFO second output");
    end
    // Expected: 70 + 1 = 71 = 0x47, tag=2
    if (out_data !== {4'h2, 32'h0000_0047}) begin : check_t2b_data
      $fatal(1, "Test 2b: expected {tag=2, 0x47} (FIFO second=70), got 0x%09h", out_data);
    end
    pass_count = pass_count + 1;
    $display("[%0t] PASS test 2b: FIFO second 70+1=71", $time);

    in0_valid = 0;
    in1_valid = 0;
    repeat (4) @(posedge clk); #1;

    // ================================================================
    // Config B: dual-register read (reset first to clear state)
    //   Insn 0 (tag=1): op0=in, op1=in -> reg(0)   [write reg 0]
    //   Insn 1 (tag=3): op0=in, op1=in -> reg(1)   [write reg 1]
    //   Insn 2 (tag=4): op0=reg(0), op1=reg(1) -> out(tag=4)
    //   Insn 3: invalid
    // ================================================================
    rst_n = 0;
    in0_valid = 0;
    in1_valid = 0;
    t0_cfg_data = '0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk); #1;

    insn0 = make_insn(1, 4'd1, 0, 0, 0, 0, 1, 0, 4'd0);  // -> reg(0)
    insn1 = make_insn(1, 4'd3, 0, 0, 0, 0, 1, 1, 4'd0);  // -> reg(1)
    insn2 = make_insn(1, 4'd4, 1, 1, 1, 0, 0, 0, 4'd4);  // reg(0)+reg(1)->out
    insn3 = 15'd0;
    t0_cfg_data = {insn3, insn2, insn1, insn0};
    @(posedge clk); #1;

    // ---- Test 3: Two-register simultaneous read ----
    // Write reg(0): 10 + 20 = 30
    in0_valid = 1;
    in1_valid = 1;
    in0_data  = {4'h1, 32'h0000_000A};
    in1_data  = {4'h1, 32'h0000_0014};
    out_ready = 1;

    in0_seen = 1'b0;
    in1_seen = 1'b0;
    cycle_count = 0;
    while (!(in0_seen && in1_seen) && cycle_count < 20) begin : wait_wr3a_accept
      @(posedge clk); #1;
      if (in0_valid && in0_ready) in0_seen = 1'b1;
      if (in1_valid && in1_ready) in1_seen = 1'b1;
      cycle_count = cycle_count + 1;
    end
    if (!(in0_seen && in1_seen)) begin : check_wr3a_accept
      $fatal(1, "Test 3a: timeout waiting reg0 write accepted");
    end
    in0_valid = 0;
    in1_valid = 0;
    @(posedge clk); #1;
    repeat (4) @(posedge clk); #1;

    // Write reg(1): 20 + 30 = 50
    in0_valid = 1;
    in1_valid = 1;
    in0_data  = {4'h3, 32'h0000_0014};
    in1_data  = {4'h3, 32'h0000_001E};
    out_ready = 1;

    in0_seen = 1'b0;
    in1_seen = 1'b0;
    cycle_count = 0;
    while (!(in0_seen && in1_seen) && cycle_count < 20) begin : wait_wr3b_accept
      @(posedge clk); #1;
      if (in0_valid && in0_ready) in0_seen = 1'b1;
      if (in1_valid && in1_ready) in1_seen = 1'b1;
      cycle_count = cycle_count + 1;
    end
    if (!(in0_seen && in1_seen)) begin : check_wr3b_accept
      $fatal(1, "Test 3b: timeout waiting reg1 write accepted");
    end
    in0_valid = 0;
    in1_valid = 0;
    @(posedge clk); #1;

    // Fire insn 2: both operands are register-sourced, so no input token is needed.
    in0_valid = 0;
    in1_valid = 0;
    in0_data  = '0;
    in1_data  = '0;
    out_ready = 1;

    t3_seen = 1'b0;
    t3_data = '0;
    if (out_valid) begin : t3_immediate
      t3_seen = 1'b1;
      t3_data = out_data;
    end

    cycle_count = 0;
    while (!t3_seen && cycle_count < 30) begin : wait_dual_out
      @(posedge clk); #1;
      if (out_valid) begin : t3_capture
        t3_seen = 1'b1;
        t3_data = out_data;
      end
      cycle_count = cycle_count + 1;
    end

    if (!t3_seen) begin : check_t3_valid
      $fatal(1, "Test 3: timeout waiting for dual-register output");
    end
    // Expected: 30 + 50 = 80 = 0x50, tag=4
    if (t3_data !== {4'h4, 32'h0000_0050}) begin : check_t3_data
      $fatal(1, "Test 3: expected {tag=4, 0x50}, got 0x%09h", t3_data);
    end
    pass_count = pass_count + 1;
    $display("[%0t] PASS test 3: dual-reg read: 30+50=80", $time);

    in0_valid = 0;
    in1_valid = 0;
    @(posedge clk); #1;

    // ---- Test 4: Final error check ----
    if (error_valid !== 0) begin : check_final_error
      $fatal(1, "Test 4: unexpected error_valid=1, code=%0d", error_code);
    end
    pass_count = pass_count + 1;
    $display("[%0t] PASS test 4: no errors throughout", $time);

    $display("PASS: tb_temporal_pe_reg_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : watchdog
    #200000;
    $fatal(1, "watchdog timeout");
  end

endmodule
