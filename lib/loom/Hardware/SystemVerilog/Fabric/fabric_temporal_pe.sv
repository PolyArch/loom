//===-- fabric_temporal_pe.sv - Temporal PE module --------------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Multi-FU temporal Processing Element with tag-based instruction dispatch.
// Inputs are tag-matched against an instruction memory to select FU and
// operand sources. Supports internal registers with FIFO buffering.
//
// Errors:
//   CFG_TEMPORAL_PE_DUP_TAG       - Duplicate tags in instruction_mem
//   CFG_TEMPORAL_PE_ILLEGAL_REG   - Register index >= NUM_REGISTERS
//   CFG_TEMPORAL_PE_REG_TAG_NONZERO - res_tag != 0 when writing register
//   RT_TEMPORAL_PE_NO_MATCH       - Input tag matches no instruction
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module fabric_temporal_pe #(
    parameter int NUM_INPUTS        = 2,
    parameter int NUM_OUTPUTS       = 1,
    parameter int DATA_WIDTH        = 32,
    parameter int TAG_WIDTH         = 4,
    parameter int NUM_FU_TYPES      = 1,
    parameter int NUM_REGISTERS     = 0,
    parameter int NUM_INSTRUCTIONS  = 2,
    parameter int REG_FIFO_DEPTH    = 0,
    localparam int PAYLOAD_WIDTH    = DATA_WIDTH + TAG_WIDTH,
    localparam int SAFE_PW          = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1,
    localparam int SAFE_DW          = (DATA_WIDTH > 0) ? DATA_WIDTH : 1,
    // Instruction memory config width
    localparam int REG_BITS         = (NUM_REGISTERS > 0) ? (1 + $clog2(NUM_REGISTERS > 1 ? NUM_REGISTERS : 2)) : 0,
    localparam int FU_SEL_BITS      = (NUM_FU_TYPES > 1) ? $clog2(NUM_FU_TYPES) : 0,
    localparam int RES_BITS         = (NUM_REGISTERS > 0) ? (1 + $clog2(NUM_REGISTERS > 1 ? NUM_REGISTERS : 2)) : 0,
    localparam int RESULT_WIDTH     = RES_BITS + TAG_WIDTH,
    localparam int INSN_WIDTH       = 1 + TAG_WIDTH + FU_SEL_BITS + NUM_INPUTS * REG_BITS + NUM_OUTPUTS * RESULT_WIDTH,
    localparam int CONFIG_WIDTH     = NUM_INSTRUCTIONS * INSN_WIDTH
) (
    input  logic                       clk,
    input  logic                       rst_n,

    // Streaming inputs
    input  logic [NUM_INPUTS-1:0]      in_valid,
    output logic [NUM_INPUTS-1:0]      in_ready,
    input  logic [NUM_INPUTS-1:0][SAFE_PW-1:0] in_data,

    // Streaming outputs
    output logic [NUM_OUTPUTS-1:0]     out_valid,
    input  logic [NUM_OUTPUTS-1:0]     out_ready,
    output logic [NUM_OUTPUTS-1:0][SAFE_PW-1:0] out_data,

    // Configuration: instruction memory
    input  logic [CONFIG_WIDTH > 0 ? CONFIG_WIDTH-1 : 0 : 0] cfg_data,

    // Error output
    output logic                       error_valid,
    output logic [15:0]                error_code
);

  // -----------------------------------------------------------------------
  // Elaboration-time parameter validation
  // -----------------------------------------------------------------------
  initial begin : param_check
    if (NUM_INPUTS < 1)
      $fatal(1, "COMP_TEMPORAL_PE_NUM_INPUTS: must be >= 1");
    if (NUM_OUTPUTS < 1)
      $fatal(1, "COMP_TEMPORAL_PE_NUM_OUTPUTS: must be >= 1");
    if (TAG_WIDTH < 1)
      $fatal(1, "COMP_TEMPORAL_PE_TAG_WIDTH: must be >= 1");
    if (NUM_INSTRUCTIONS < 1)
      $fatal(1, "COMP_TEMPORAL_PE_NUM_INSTRUCTION: must be >= 1");
    if (NUM_REGISTERS > 0 && REG_FIFO_DEPTH < 1)
      $fatal(1, "COMP_TEMPORAL_PE_REG_FIFO_DEPTH: must be >= 1 when registers enabled");
  end

  // -----------------------------------------------------------------------
  // Instruction memory unpacking
  // Each instruction: {valid(1), tag(TAG_WIDTH), fu_sel, operand_src[], result_dst[]}
  // -----------------------------------------------------------------------
  logic [NUM_INSTRUCTIONS-1:0]                     insn_valid;
  logic [NUM_INSTRUCTIONS-1:0][TAG_WIDTH-1:0]      insn_tag;

  always_comb begin : unpack_insn
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_INSTRUCTIONS; iter_var0 = iter_var0 + 1) begin : per_insn
      automatic int base = iter_var0 * INSN_WIDTH;
      insn_valid[iter_var0] = cfg_data[base + INSN_WIDTH - 1];
      insn_tag[iter_var0]   = cfg_data[base + INSN_WIDTH - 2 -: TAG_WIDTH];
    end
  end

  // -----------------------------------------------------------------------
  // Tag matching: find instruction for current input tag
  // -----------------------------------------------------------------------
  logic [NUM_INPUTS-1:0][TAG_WIDTH-1:0] in_tag;
  logic [NUM_INPUTS-1:0][SAFE_DW-1:0]  in_value;
  logic                                  all_in_valid;
  logic                                  match_found;
  logic [$clog2(NUM_INSTRUCTIONS > 1 ? NUM_INSTRUCTIONS : 2)-1:0] matched_insn;

  assign all_in_valid = &in_valid;

  always_comb begin : extract_tags
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : per_input
      in_value[iter_var0] = in_data[iter_var0][DATA_WIDTH-1:0];
      in_tag[iter_var0]   = in_data[iter_var0][DATA_WIDTH +: TAG_WIDTH];
    end
  end

  always_comb begin : find_insn
    integer iter_var0;
    match_found = 1'b0;
    matched_insn = '0;
    for (iter_var0 = 0; iter_var0 < NUM_INSTRUCTIONS; iter_var0 = iter_var0 + 1) begin : search
      if (insn_valid[iter_var0] && (insn_tag[iter_var0] == in_tag[0])) begin : found
        match_found  = 1'b1;
        matched_insn = iter_var0[$clog2(NUM_INSTRUCTIONS > 1 ? NUM_INSTRUCTIONS : 2)-1:0];
      end
    end
  end

  // -----------------------------------------------------------------------
  // Compute: pass-through body (filled per-instance by exportSV)
  // -----------------------------------------------------------------------
  logic [NUM_OUTPUTS-1:0][SAFE_DW-1:0] body_result;

  // ===== BEGIN PE BODY =====
  // (replaced by exportSV based on instruction FU selection)
  // ===== END PE BODY =====

  // -----------------------------------------------------------------------
  // Output assembly with tag from instruction
  // -----------------------------------------------------------------------
  logic all_out_ready;
  assign all_out_ready = &out_ready;

  logic fire;
  assign fire = all_in_valid && match_found && all_out_ready;

  generate
    genvar go;
    for (go = 0; go < NUM_OUTPUTS; go++) begin : g_out
      assign out_valid[go] = all_in_valid && match_found;
      assign out_data[go]  = {in_tag[0], body_result[go]};
    end
    genvar gi;
    for (gi = 0; gi < NUM_INPUTS; gi++) begin : g_in_ready
      assign in_ready[gi] = fire;
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Error detection
  // -----------------------------------------------------------------------
  logic        err_detect;
  logic [15:0] err_code_comb;

  always_comb begin : err_check
    integer iter_var0, iter_var1;
    err_detect    = 1'b0;
    err_code_comb = 16'hFFFF;

    // CFG_TEMPORAL_PE_DUP_TAG: duplicate valid tags
    for (iter_var0 = 0; iter_var0 < NUM_INSTRUCTIONS; iter_var0 = iter_var0 + 1) begin : chk_dup_outer
      for (iter_var1 = iter_var0 + 1; iter_var1 < NUM_INSTRUCTIONS; iter_var1 = iter_var1 + 1) begin : chk_dup_inner
        if (insn_valid[iter_var0] && insn_valid[iter_var1] &&
            (insn_tag[iter_var0] == insn_tag[iter_var1])) begin : dup
          err_detect = 1'b1;
          if (CFG_TEMPORAL_PE_DUP_TAG < err_code_comb)
            err_code_comb = CFG_TEMPORAL_PE_DUP_TAG;
        end
      end
    end

    // RT_TEMPORAL_PE_NO_MATCH: all inputs valid but no instruction matches
    if (all_in_valid && !match_found) begin : rt_no_match
      err_detect = 1'b1;
      if (RT_TEMPORAL_PE_NO_MATCH < err_code_comb)
        err_code_comb = RT_TEMPORAL_PE_NO_MATCH;
    end

    if (!err_detect)
      err_code_comb = 16'd0;
  end

  // Error latch
  always_ff @(posedge clk or negedge rst_n) begin : error_latch
    if (!rst_n) begin : reset
      error_valid <= 1'b0;
      error_code  <= 16'd0;
    end else if (!error_valid && err_detect) begin : capture
      error_valid <= 1'b1;
      error_code  <= err_code_comb;
    end
  end

endmodule
