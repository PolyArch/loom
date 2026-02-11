//===-- tb_fabric_temporal_pe.sv - Parameterized temporal PE test -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_fabric_temporal_pe;
  parameter int NUM_INPUTS           = 2;
  parameter int NUM_OUTPUTS          = 1;
  parameter int DATA_WIDTH           = 32;
  parameter int TAG_WIDTH            = 4;
  parameter int NUM_FU_TYPES         = 1;
  parameter int NUM_REGISTERS        = 0;
  parameter int NUM_INSTRUCTIONS     = 2;
  parameter int REG_FIFO_DEPTH       = 0;
  parameter int SHARED_OPERAND_BUFFER         = 0;
  parameter int OPERAND_BUFFER_SIZE  = 0;

  localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH;
  localparam int SAFE_PW = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1;
  localparam int INSN_VALID_LSB = 0;
  localparam int INSN_TAG_LSB = 1;

  logic clk, rst_n;
  logic [NUM_INPUTS-1:0]                 in_valid;
  logic [NUM_INPUTS-1:0]                 in_ready;
  logic [NUM_INPUTS-1:0][SAFE_PW-1:0]   in_data;
  logic [NUM_OUTPUTS-1:0]               out_valid;
  logic [NUM_OUTPUTS-1:0]               out_ready;
  logic [NUM_OUTPUTS-1:0][SAFE_PW-1:0]  out_data;
  // Use a large config width placeholder (CONFIG_WIDTH is derived internally)
  logic [255:0] cfg_data;
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

    // Check 2: CFG_TEMPORAL_PE_DUP_TAG - configure two instructions with same tag
    // Instruction layout from LSB: [valid | tag | opcode | operands | results]
    // INSN_WIDTH = 1 + TAG_WIDTH + FU_SEL_BITS + NUM_INPUTS*REG_BITS + NUM_OUTPUTS*RESULT_WIDTH
    // For default params (NR=0): REG_BITS=0, RES_BITS=0, RESULT_WIDTH=TAG_WIDTH
    // INSN_WIDTH = 1 + TAG_WIDTH + FU_SEL_BITS + TAG_WIDTH
    // Insn 0: valid=1, tag=3; Insn 1: valid=1, tag=3 (duplicate)
    cfg_data = '0;
    // Insn 0
    cfg_data[0 * dut.INSN_WIDTH + INSN_VALID_LSB] = 1'b1;
    cfg_data[0 * dut.INSN_WIDTH + INSN_TAG_LSB +: TAG_WIDTH] = TAG_WIDTH'(3);
    // Insn 1
    cfg_data[1 * dut.INSN_WIDTH + INSN_VALID_LSB] = 1'b1;
    cfg_data[1 * dut.INSN_WIDTH + INSN_TAG_LSB +: TAG_WIDTH] = TAG_WIDTH'(3);
    @(posedge clk);
    @(posedge clk);
    if (error_valid !== 1'b1) begin : check_dup_tag
      $fatal(1, "expected CFG_TEMPORAL_PE_DUP_TAG error");
    end
    if (error_code !== CFG_TEMPORAL_PE_DUP_TAG) begin : check_dup_code
      $fatal(1, "wrong error code for dup tag: got %0d", error_code);
    end
    pass_count = pass_count + 1;

    // Check 3: RT_TEMPORAL_PE_NO_MATCH - send tag not in instruction table
    rst_n = 0;
    in_valid = '0;
    cfg_data = '0;
    repeat (2) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    // Configure one valid instruction with tag=1
    cfg_data[0 * dut.INSN_WIDTH + INSN_VALID_LSB] = 1'b1;
    cfg_data[0 * dut.INSN_WIDTH + INSN_TAG_LSB +: TAG_WIDTH] = TAG_WIDTH'(1);
    // Send inputs with tag=7 (no match)
    in_data = '0;
    in_data[0][DATA_WIDTH +: TAG_WIDTH] = TAG_WIDTH'(7);
    if (NUM_INPUTS > 1) begin : set_in1
      in_data[1][DATA_WIDTH +: TAG_WIDTH] = TAG_WIDTH'(7);
    end
    in_valid = {NUM_INPUTS{1'b1}};
    @(posedge clk);
    @(posedge clk);
    if (error_valid !== 1'b1) begin : check_no_match
      $fatal(1, "expected RT_TEMPORAL_PE_NO_MATCH error");
    end
    if (error_code !== RT_TEMPORAL_PE_NO_MATCH) begin : check_no_match_code
      $fatal(1, "wrong error code for no match: got %0d", error_code);
    end
    pass_count = pass_count + 1;
    in_valid = '0;

    $display("PASS: tb_fabric_temporal_pe NI=%0d NO=%0d DW=%0d TW=%0d (%0d checks)",
             NUM_INPUTS, NUM_OUTPUTS, DATA_WIDTH, TAG_WIDTH, pass_count);
    $finish;
  end

  initial begin : timeout
    #10000;
    $fatal(1, "TIMEOUT");
  end
endmodule
