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
  parameter int NUM_INPUTS           = 2;
  parameter int NUM_OUTPUTS          = 1;
  parameter int DATA_WIDTH           = 32;
  parameter int TAG_WIDTH            = 4;
  parameter int NUM_FU_TYPES         = 1;
  parameter int NUM_REGISTERS        = 3;
  parameter int NUM_INSTRUCTIONS     = 2;
  parameter int REG_FIFO_DEPTH       = 2;
  parameter int SHARED_OPERAND_BUFFER         = 0;
  parameter int OPERAND_BUFFER_SIZE  = 0;

  localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH;
  localparam int SAFE_PW = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1;
  localparam int REG_BITS = 1 + $clog2(NUM_REGISTERS > 1 ? NUM_REGISTERS : 2);
  localparam int FU_SEL_BITS = (NUM_FU_TYPES > 1) ? $clog2(NUM_FU_TYPES) : 0;
  localparam int RES_BITS = REG_BITS;
  localparam int RESULT_WIDTH = RES_BITS + TAG_WIDTH;
  localparam int INSN_WIDTH =
      1 + TAG_WIDTH + FU_SEL_BITS + NUM_INPUTS * REG_BITS + NUM_OUTPUTS * RESULT_WIDTH;
  localparam int INSN_VALID_LSB = 0;
  localparam int INSN_TAG_LSB = INSN_VALID_LSB + 1;
  localparam int INSN_OPERANDS_LSB = INSN_TAG_LSB + TAG_WIDTH + FU_SEL_BITS;
  localparam int INSN_RESULTS_LSB = INSN_OPERANDS_LSB + NUM_INPUTS * REG_BITS;

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
    .REG_FIFO_DEPTH(REG_FIFO_DEPTH),
    .SHARED_OPERAND_BUFFER(SHARED_OPERAND_BUFFER),
    .OPERAND_BUFFER_SIZE(OPERAND_BUFFER_SIZE)
  ) dut (
    .clk(clk), .rst_n(rst_n),
    .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data),
    .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
    .cfg_data(cfg_data[dut.CONFIG_WIDTH > 0 ? dut.CONFIG_WIDTH-1 : 0 : 0]),
    .error_valid(error_valid), .error_code(error_code)
  );

  // For NUM_REGISTERS=3: REG_BITS = 3, RES_BITS = 3, RESULT_WIDTH = 7, INSN_WIDTH = 18.
  // Full instruction layout from LSB:
  //   valid(1) | tag(4) | fu_sel(0) | op0(3) | op1(3) | result0(7)
  // Operand field (3 bits): reg_idx[1:0] (LSB), is_reg (MSB)
  // Result field (7 bits):  res_tag[3:0] (LSB), reg_idx[1:0], is_reg (MSB)

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
    cfg_data[0 * INSN_WIDTH + INSN_VALID_LSB] = 1'b1;
    cfg_data[0 * INSN_WIDTH + INSN_TAG_LSB +: TAG_WIDTH] = TAG_WIDTH'(1);
    // operand0 = reg(3): is_reg=1, reg_idx=3
    cfg_data[0 * INSN_WIDTH + INSN_OPERANDS_LSB + REG_BITS - 1] = 1'b1;
    cfg_data[0 * INSN_WIDTH + INSN_OPERANDS_LSB +: (REG_BITS - 1)] = (REG_BITS-1)'(3);
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
    cfg_data[0 * INSN_WIDTH + INSN_VALID_LSB] = 1'b1;
    cfg_data[0 * INSN_WIDTH + INSN_TAG_LSB +: TAG_WIDTH] = TAG_WIDTH'(2);
    // result0: is_reg=1, reg_idx=0, res_tag=5 (nonzero -> error)
    cfg_data[0 * INSN_WIDTH + INSN_RESULTS_LSB + RESULT_WIDTH - 1] = 1'b1;
    cfg_data[0 * INSN_WIDTH + INSN_RESULTS_LSB + TAG_WIDTH +: (RES_BITS - 1)] = (RES_BITS-1)'(0);
    cfg_data[0 * INSN_WIDTH + INSN_RESULTS_LSB +: TAG_WIDTH] = TAG_WIDTH'(5);
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
