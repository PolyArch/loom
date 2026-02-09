//===-- tb_fabric_temporal_pe_reg.sv - Temporal PE register tests -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Tests CFG_TEMPORAL_PE_ILLEGAL_REG and CFG_TEMPORAL_PE_REG_TAG_NONZERO
// errors that require NUM_REGISTERS > 0. Uses NUM_REGISTERS=3 so the
// register index field (2 bits) can represent OOB value 3.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_fabric_temporal_pe_reg;
  parameter int NUM_INPUTS       = 2;
  parameter int NUM_OUTPUTS      = 1;
  parameter int DATA_WIDTH       = 32;
  parameter int TAG_WIDTH        = 4;
  parameter int NUM_FU_TYPES     = 1;
  parameter int NUM_REGISTERS    = 3;
  parameter int NUM_INSTRUCTIONS = 2;
  parameter int REG_FIFO_DEPTH   = 2;

  localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH;
  localparam int SAFE_PW = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1;

  logic clk, rst_n;
  logic [NUM_INPUTS-1:0]                 in_valid;
  logic [NUM_INPUTS-1:0]                 in_ready;
  logic [NUM_INPUTS-1:0][SAFE_PW-1:0]   in_data;
  logic [NUM_OUTPUTS-1:0]               out_valid;
  logic [NUM_OUTPUTS-1:0]               out_ready;
  logic [NUM_OUTPUTS-1:0][SAFE_PW-1:0]  out_data;
  // Use a large config width placeholder (CONFIG_WIDTH is derived internally)
  logic [511:0] cfg_data;
  logic        error_valid;
  logic [15:0] error_code;

  fabric_temporal_pe #(
    .NUM_INPUTS(NUM_INPUTS),
    .NUM_OUTPUTS(NUM_OUTPUTS),
    .DATA_WIDTH(DATA_WIDTH),
    .TAG_WIDTH(TAG_WIDTH),
    .NUM_FU_TYPES(NUM_FU_TYPES),
    .NUM_REGISTERS(NUM_REGISTERS),
    .NUM_INSTRUCTIONS(NUM_INSTRUCTIONS),
    .REG_FIFO_DEPTH(REG_FIFO_DEPTH)
  ) dut (
    .clk(clk), .rst_n(rst_n),
    .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data),
    .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
    .cfg_data(cfg_data[dut.CONFIG_WIDTH > 0 ? dut.CONFIG_WIDTH-1 : 0 : 0]),
    .error_valid(error_valid), .error_code(error_code)
  );

  // For NUM_REGISTERS=3: REG_BITS = 1 + clog2(3) = 3, RES_BITS = 3
  // RESULT_WIDTH = RES_BITS + TAG_WIDTH = 3 + 4 = 7
  // INSN_WIDTH = 1 + TAG_WIDTH + FU_SEL_BITS + NUM_INPUTS*REG_BITS + NUM_OUTPUTS*RESULT_WIDTH
  //            = 1 + 4 + 0 + 2*3 + 1*7 = 18
  //
  // Instruction bit layout from res_base (per result):
  //   [res_tag(TAG_WIDTH=4)] [reg_idx(RES_BITS-1=2)] [is_reg(1)]  -- but check:
  //   Actually from code: is_reg = cfg_data[res_base + RESULT_WIDTH - 1]
  //   reg_idx = cfg_data[res_base + TAG_WIDTH +: (RES_BITS - 1)]
  //   res_tag = cfg_data[res_base +: TAG_WIDTH]
  //   So from LSB: res_tag(4), reg_idx(2), is_reg(1)  = 7 bits = RESULT_WIDTH
  //
  // Operand bit layout from op_base (per operand):
  //   is_reg = cfg_data[op_base + REG_BITS - 1]
  //   reg_idx = cfg_data[op_base +: (REG_BITS - 1)]
  //   From LSB: reg_idx(2), is_reg(1)  = 3 bits = REG_BITS
  //
  // Full instruction from LSB:
  //   result0(7) | op0(3) | op1(3) | fu_sel(0) | tag(4) | valid(1) = 18 bits

  initial begin : clk_gen
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin : test
    integer pass_count;
    pass_count = 0;
    rst_n = 0;
    in_valid = '0;
    out_ready = '1;
    in_data = '0;
    cfg_data = '0;

    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // Check 1: no error after reset
    if (error_valid !== 1'b0) begin : check_reset
      $fatal(1, "error_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    // Check 2: CFG_TEMPORAL_PE_ILLEGAL_REG - operand register index OOB
    // Insn 0: valid=1, tag=1, operand 0 = {is_reg=1, reg_idx=3} (3 >= NUM_REGISTERS=3)
    cfg_data = '0;
    // valid bit at INSN_WIDTH-1 = 17
    cfg_data[17] = 1'b1;
    // tag at bits [16:13] (INSN_WIDTH-2 downto INSN_WIDTH-1-TAG_WIDTH)
    cfg_data[16:13] = TAG_WIDTH'(1);
    // operand 0 at base: result0(7 bits) = bits [6:0], then op0 starts at bit 7
    // op0: bits [9:7], where bit 9 = is_reg, bits [8:7] = reg_idx
    cfg_data[9] = 1'b1;      // is_reg = 1
    cfg_data[8:7] = 2'd3;    // reg_idx = 3 (OOB: >= NUM_REGISTERS=3)
    @(posedge clk);
    @(posedge clk);
    if (error_valid !== 1'b1) begin : check_illegal_reg
      $fatal(1, "expected CFG_TEMPORAL_PE_ILLEGAL_REG error");
    end
    if (error_code !== CFG_TEMPORAL_PE_ILLEGAL_REG) begin : check_illegal_reg_code
      $fatal(1, "wrong error code for illegal_reg: got %0d", error_code);
    end
    pass_count = pass_count + 1;

    // Check 3: CFG_TEMPORAL_PE_REG_TAG_NONZERO - result writes register with nonzero tag
    rst_n = 0;
    in_valid = '0;
    cfg_data = '0;
    repeat (2) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    // Insn 0: valid=1, tag=2
    cfg_data[17] = 1'b1;
    cfg_data[16:13] = TAG_WIDTH'(2);
    // result 0 at bits [6:0]: res_tag(4), reg_idx(2), is_reg(1)
    // Set is_reg=1 at bit 6
    cfg_data[6] = 1'b1;
    // reg_idx=0 at bits [5:4]
    cfg_data[5:4] = 2'd0;
    // res_tag=5 at bits [3:0] (nonzero -> error)
    cfg_data[3:0] = 4'd5;
    @(posedge clk);
    @(posedge clk);
    if (error_valid !== 1'b1) begin : check_reg_tag_nz
      $fatal(1, "expected CFG_TEMPORAL_PE_REG_TAG_NONZERO error");
    end
    if (error_code !== CFG_TEMPORAL_PE_REG_TAG_NONZERO) begin : check_reg_tag_nz_code
      $fatal(1, "wrong error code for reg_tag_nz: got %0d", error_code);
    end
    pass_count = pass_count + 1;

    $display("PASS: tb_fabric_temporal_pe_reg NI=%0d NO=%0d NR=%0d (%0d checks)",
             NUM_INPUTS, NUM_OUTPUTS, NUM_REGISTERS, pass_count);
    $finish;
  end

  initial begin : timeout
    #10000;
    $fatal(1, "TIMEOUT");
  end
endmodule
