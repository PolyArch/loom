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
  // Input tag/value extraction
  // -----------------------------------------------------------------------
  logic [NUM_INPUTS-1:0][TAG_WIDTH-1:0] in_tag;
  logic [NUM_INPUTS-1:0][SAFE_DW-1:0]  in_value;

  always_comb begin : extract_tags
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : per_input
      in_value[iter_var0] = in_data[iter_var0][DATA_WIDTH-1:0];
      in_tag[iter_var0]   = in_data[iter_var0][DATA_WIDTH +: TAG_WIDTH];
    end
  end

  // -----------------------------------------------------------------------
  // Operand buffer (Mode A: per-instruction buffer)
  // Each instruction slot has NUM_INPUTS entries: {op_valid, op_value}
  // For register-sourced operands, the buffer is pre-filled from register FIFOs.
  // -----------------------------------------------------------------------
  localparam int INSN_IDX_W = $clog2(NUM_INSTRUCTIONS > 1 ? NUM_INSTRUCTIONS : 2);
  // Sequential operand buffer (input-sourced operands)
  logic [NUM_INSTRUCTIONS-1:0][NUM_INPUTS-1:0] op_buf_valid;
  logic [NUM_INSTRUCTIONS-1:0][NUM_INPUTS-1:0][SAFE_DW-1:0] op_buf_value;
  // Effective operand state: sequential buffer overridden by register FIFO
  logic [NUM_INSTRUCTIONS-1:0][NUM_INPUTS-1:0] op_valid;
  logic [NUM_INSTRUCTIONS-1:0][NUM_INPUTS-1:0][SAFE_DW-1:0] op_value;

  // Per-input tag match: find instruction slot for each input's tag
  logic [NUM_INPUTS-1:0]                       in_match;
  logic [NUM_INPUTS-1:0][INSN_IDX_W-1:0]      in_matched_slot;

  always_comb begin : per_input_match
    integer iter_var0, iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : per_in
      in_match[iter_var0] = 1'b0;
      in_matched_slot[iter_var0] = '0;
      for (iter_var1 = 0; iter_var1 < NUM_INSTRUCTIONS; iter_var1 = iter_var1 + 1) begin : search
        if (insn_valid[iter_var1] && (insn_tag[iter_var1] == in_tag[iter_var0])) begin : found
          in_match[iter_var0] = 1'b1;
          in_matched_slot[iter_var0] = iter_var1[INSN_IDX_W-1:0];
        end
      end
    end
  end

  // Accept input when: valid, tag matches an instruction, and buffer slot
  // for that instruction/input is empty. Only checked against sequential buffer.
  logic [NUM_INPUTS-1:0] in_accept;
  // Instruction ready to fire: all operands buffered
  logic                  match_found;
  logic [INSN_IDX_W-1:0] matched_insn;
  logic                  insn_fire_ready;

  // Find the first instruction slot where all operands are ready
  always_comb begin : find_ready_insn
    integer iter_var0, iter_var1;
    match_found = 1'b0;
    matched_insn = '0;
    insn_fire_ready = 1'b0;
    for (iter_var0 = 0; iter_var0 < NUM_INSTRUCTIONS; iter_var0 = iter_var0 + 1) begin : per_slot
      if (insn_valid[iter_var0] && !match_found) begin : check_ready
        automatic logic all_ready = 1'b1;
        for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1 = iter_var1 + 1) begin : per_op
          if (!op_valid[iter_var0][iter_var1]) begin : not_ready
            all_ready = 1'b0;
          end
        end
        if (all_ready) begin : ready
          match_found = 1'b1;
          matched_insn = iter_var0[INSN_IDX_W-1:0];
          insn_fire_ready = 1'b1;
        end
      end
    end
  end

  // -----------------------------------------------------------------------
  // FU launch one-shot: prevent FU replay during multi-cycle latency
  // fu_launch pulses high for exactly one cycle when an instruction fires.
  // fu_busy stays high until body_valid indicates completion.
  // -----------------------------------------------------------------------
  logic fu_busy;
  logic fu_launch;
  assign fu_launch = insn_fire_ready && !fu_busy;

  always_ff @(posedge clk or negedge rst_n) begin : fu_busy_reg
    if (!rst_n) begin : reset
      fu_busy <= 1'b0;
    end else begin : tick
      if (fu_launch) begin : launch
        fu_busy <= 1'b1;
      end else if (fu_busy && body_valid) begin : complete
        fu_busy <= 1'b0;
      end
    end
  end

  // -----------------------------------------------------------------------
  // Compute: pass-through body (filled per-instance by exportSV)
  // body_valid: driven by generated body; 1 when FU result is ready.
  // fu_operands feeds from operand buffer for the firing instruction.
  // fu_launch is the one-shot trigger for the FU.
  // -----------------------------------------------------------------------
  logic [NUM_OUTPUTS-1:0][SAFE_DW-1:0] body_result;
  logic body_valid;

  // Select operand buffer values for the firing instruction
  logic [NUM_INPUTS-1:0][SAFE_DW-1:0] fu_operands;
  always_comb begin : select_operands
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : per_in
      fu_operands[iter_var0] = op_value[matched_insn][iter_var0];
    end
  end

  // ===== BEGIN PE BODY =====
  // (replaced by exportSV based on instruction FU selection)
  assign body_valid = 1'b1;
  // ===== END PE BODY =====

  // -----------------------------------------------------------------------
  // Output assembly with tag from instruction result fields
  // -----------------------------------------------------------------------
  logic all_out_ready;
  assign all_out_ready = &out_ready;

  // fire: instruction commits - operand buffer cleared, outputs driven,
  // register writes occur. Requires FU completion and output readiness.
  logic fire;
  assign fire = insn_fire_ready && body_valid && all_out_ready && (fu_busy || fu_launch);

  // Extract per-output result tag from the matched instruction.
  logic [NUM_OUTPUTS-1:0][TAG_WIDTH-1:0] out_res_tag;

  always_comb begin : extract_res_tag
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_OUTPUTS; iter_var0 = iter_var0 + 1) begin : per_out
      out_res_tag[iter_var0] = cfg_data[matched_insn * INSN_WIDTH + iter_var0 * RESULT_WIDTH +: TAG_WIDTH];
    end
  end

  // Output valid/data: only for results that are NOT register-writes.
  // When NUM_REGISTERS=0, all results go to outputs.
  generate
    genvar go;
    for (go = 0; go < NUM_OUTPUTS; go++) begin : g_out
      assign out_valid[go] = fire;
      assign out_data[go]  = {out_res_tag[go], body_result[go]};
    end
  endgenerate

  // Input ready: accept when tag matches and buffer slot is available
  always_comb begin : gen_in_ready
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : per_in
      in_accept[iter_var0] = in_valid[iter_var0] && in_match[iter_var0] &&
                             !op_buf_valid[in_matched_slot[iter_var0]][iter_var0];
      in_ready[iter_var0] = in_accept[iter_var0];
    end
  end

  // -----------------------------------------------------------------------
  // Register file FIFOs (generate-time guard for NUM_REGISTERS > 0)
  // Each register is a FIFO with depth REG_FIFO_DEPTH.
  // Writing enqueues; reading dequeues (after all readers consume).
  // -----------------------------------------------------------------------
  localparam int SAFE_REG = (NUM_REGISTERS > 0) ? NUM_REGISTERS : 1;
  localparam int SAFE_RFIFO = (REG_FIFO_DEPTH > 0) ? REG_FIFO_DEPTH : 1;
  localparam int RFIFO_IDX_W = $clog2(SAFE_RFIFO > 1 ? SAFE_RFIFO : 2);
  localparam int SAFE_REG_IDX_W = (REG_BITS > 1) ? (REG_BITS - 1) : 1;

  // Register FIFO storage (always declared; only used when NUM_REGISTERS > 0)
  logic [SAFE_REG-1:0][SAFE_RFIFO-1:0][SAFE_DW-1:0] reg_fifo_data;
  logic [SAFE_REG-1:0][RFIFO_IDX_W:0]                reg_fifo_cnt;
  logic [SAFE_REG-1:0][RFIFO_IDX_W-1:0]              reg_fifo_wr_ptr;
  logic [SAFE_REG-1:0][RFIFO_IDX_W-1:0]              reg_fifo_rd_ptr;
  logic [SAFE_REG-1:0]                                reg_fifo_empty;
  logic [SAFE_REG-1:0]                                reg_fifo_full;

  generate
    if (NUM_REGISTERS > 0) begin : g_reg_fifo
      genvar gri;
      for (gri = 0; gri < NUM_REGISTERS; gri++) begin : g_per_reg
        assign reg_fifo_empty[gri] = (reg_fifo_cnt[gri] == '0);
        assign reg_fifo_full[gri]  = (reg_fifo_cnt[gri] == SAFE_RFIFO[RFIFO_IDX_W:0]);
      end

      // Register write: on fire, if res_is_reg=1 for any output, enqueue
      // Register read: on fire, operands sourced from register dequeue
      // Both handled in the reg_fifo_update block below.

      // Decode res_is_reg and res_reg_idx for the matched instruction
      logic [NUM_OUTPUTS-1:0] res_is_reg;
      logic [NUM_OUTPUTS-1:0][SAFE_REG_IDX_W-1:0] res_reg_idx;

      always_comb begin : decode_res
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < NUM_OUTPUTS; iter_var0 = iter_var0 + 1) begin : per_out
          automatic int res_base = matched_insn * INSN_WIDTH + iter_var0 * RESULT_WIDTH;
          res_is_reg[iter_var0] = cfg_data[res_base + RESULT_WIDTH - 1];
          res_reg_idx[iter_var0] = cfg_data[res_base + TAG_WIDTH +: (RES_BITS - 1)];
        end
      end

      // Decode op_is_reg and op_reg_idx for the matched instruction
      logic [NUM_INPUTS-1:0] op_is_reg;
      logic [NUM_INPUTS-1:0][SAFE_REG_IDX_W-1:0] op_reg_idx;

      always_comb begin : decode_op
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : per_in
          automatic int op_base = matched_insn * INSN_WIDTH + NUM_OUTPUTS * RESULT_WIDTH + iter_var0 * REG_BITS;
          op_is_reg[iter_var0] = cfg_data[op_base + REG_BITS - 1];
          op_reg_idx[iter_var0] = cfg_data[op_base +: (REG_BITS - 1)];
        end
      end

      // Register FIFO update
      always_ff @(posedge clk or negedge rst_n) begin : reg_fifo_update
        integer iter_var0, iter_var1;
        if (!rst_n) begin : reset
          for (iter_var0 = 0; iter_var0 < NUM_REGISTERS; iter_var0 = iter_var0 + 1) begin : clr_reg
            reg_fifo_cnt[iter_var0]    <= '0;
            reg_fifo_wr_ptr[iter_var0] <= '0;
            reg_fifo_rd_ptr[iter_var0] <= '0;
          end
        end else begin : tick
          if (fire) begin : commit
            // Enqueue: write results to register FIFOs
            for (iter_var0 = 0; iter_var0 < NUM_OUTPUTS; iter_var0 = iter_var0 + 1) begin : wr_reg
              if (res_is_reg[iter_var0] && !reg_fifo_full[res_reg_idx[iter_var0]]) begin : enq
                reg_fifo_data[res_reg_idx[iter_var0]][reg_fifo_wr_ptr[res_reg_idx[iter_var0]]] <= body_result[iter_var0];
                reg_fifo_wr_ptr[res_reg_idx[iter_var0]] <=
                  (reg_fifo_wr_ptr[res_reg_idx[iter_var0]] == RFIFO_IDX_W'(SAFE_RFIFO - 1))
                    ? '0 : (reg_fifo_wr_ptr[res_reg_idx[iter_var0]] + RFIFO_IDX_W'(1));
                reg_fifo_cnt[res_reg_idx[iter_var0]] <=
                  reg_fifo_cnt[res_reg_idx[iter_var0]] + (RFIFO_IDX_W+1)'(1);
              end
            end
            // Dequeue: consume register operands
            for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : rd_reg
              if (op_is_reg[iter_var0] && !reg_fifo_empty[op_reg_idx[iter_var0]]) begin : deq
                reg_fifo_rd_ptr[op_reg_idx[iter_var0]] <=
                  (reg_fifo_rd_ptr[op_reg_idx[iter_var0]] == RFIFO_IDX_W'(SAFE_RFIFO - 1))
                    ? '0 : (reg_fifo_rd_ptr[op_reg_idx[iter_var0]] + RFIFO_IDX_W'(1));
                reg_fifo_cnt[op_reg_idx[iter_var0]] <=
                  reg_fifo_cnt[op_reg_idx[iter_var0]] - (RFIFO_IDX_W+1)'(1);
              end
            end
          end
        end
      end

      // Fill register-sourced operands into op_valid/op_value combinationally.
      // For register operands, op_valid reflects register FIFO non-empty.
      // This is done in the operand buffer update block via op_is_reg checks.
    end else begin : g_no_reg_fifo
      // No register file - FIFOs are unused, tie off empty/full
      genvar gri_no;
      for (gri_no = 0; gri_no < SAFE_REG; gri_no++) begin : g_tie
        assign reg_fifo_empty[gri_no] = 1'b1;
        assign reg_fifo_full[gri_no]  = 1'b0;
      end
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Operand buffer update
  // For input-sourced operands (op_is_reg=0 or NUM_REGISTERS=0):
  //   Buffers arriving input values; cleared on fire.
  // For register-sourced operands (op_is_reg=1, NUM_REGISTERS>0):
  //   op_valid reflects register FIFO non-empty (combinational, not buffered).
  // -----------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin : op_buf_update
    integer iter_var0, iter_var1;
    if (!rst_n) begin : reset
      for (iter_var0 = 0; iter_var0 < NUM_INSTRUCTIONS; iter_var0 = iter_var0 + 1) begin : clr_insn
        for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1 = iter_var1 + 1) begin : clr_op
          op_buf_valid[iter_var0][iter_var1] <= 1'b0;
          op_buf_value[iter_var0][iter_var1] <= '0;
        end
      end
    end else begin : update
      // Clear fired instruction's buffer (input-sourced operands)
      if (fire) begin : clear_fired
        for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1 = iter_var1 + 1) begin : clr_op
          op_buf_valid[matched_insn][iter_var1] <= 1'b0;
        end
      end
      // Buffer arriving operands (input-sourced)
      for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : buf_in
        if (in_accept[iter_var0]) begin : accept
          op_buf_valid[in_matched_slot[iter_var0]][iter_var0] <= 1'b1;
          op_buf_value[in_matched_slot[iter_var0]][iter_var0] <= in_value[iter_var0];
        end
      end
    end
  end

  // Effective operand merge: combine sequential buffer with register FIFO state.
  // For register-sourced operands (op_is_reg=1), override with FIFO non-empty/data.
  // For input-sourced operands, pass through the sequential buffer.
  generate
    if (NUM_REGISTERS > 0) begin : g_reg_op_override
      always_comb begin : eff_op_merge
        integer iter_var0, iter_var1;
        for (iter_var0 = 0; iter_var0 < NUM_INSTRUCTIONS; iter_var0 = iter_var0 + 1) begin : per_insn
          for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1 = iter_var1 + 1) begin : per_op
            // Default: pass through sequential buffer
            op_valid[iter_var0][iter_var1] = op_buf_valid[iter_var0][iter_var1];
            op_value[iter_var0][iter_var1] = op_buf_value[iter_var0][iter_var1];
            if (insn_valid[iter_var0]) begin : valid_insn
              automatic int op_base = iter_var0 * INSN_WIDTH + NUM_OUTPUTS * RESULT_WIDTH + iter_var1 * REG_BITS;
              if (cfg_data[op_base + REG_BITS - 1]) begin : is_reg_op
                // Register-sourced: op_valid = FIFO non-empty
                automatic int ridx = cfg_data[op_base +: (REG_BITS - 1)];
                op_valid[iter_var0][iter_var1] = !reg_fifo_empty[ridx];
                op_value[iter_var0][iter_var1] = reg_fifo_data[ridx][reg_fifo_rd_ptr[ridx]];
              end
            end
          end
        end
      end
    end else begin : g_no_reg_override
      always_comb begin : eff_op_passthrough
        integer iter_var0, iter_var1;
        for (iter_var0 = 0; iter_var0 < NUM_INSTRUCTIONS; iter_var0 = iter_var0 + 1) begin : per_insn
          for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1 = iter_var1 + 1) begin : per_op
            op_valid[iter_var0][iter_var1] = op_buf_valid[iter_var0][iter_var1];
            op_value[iter_var0][iter_var1] = op_buf_value[iter_var0][iter_var1];
          end
        end
      end
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Error detection
  // -----------------------------------------------------------------------
  logic        err_dup_tag;
  logic        err_no_match;
  logic        err_illegal_reg;
  logic        err_reg_tag_nz;

  // CFG_TEMPORAL_PE_DUP_TAG: duplicate valid tags
  always_comb begin : chk_dup_tag
    integer iter_var0, iter_var1;
    err_dup_tag = 1'b0;
    for (iter_var0 = 0; iter_var0 < NUM_INSTRUCTIONS; iter_var0 = iter_var0 + 1) begin : outer
      for (iter_var1 = iter_var0 + 1; iter_var1 < NUM_INSTRUCTIONS; iter_var1 = iter_var1 + 1) begin : inner
        if (insn_valid[iter_var0] && insn_valid[iter_var1] &&
            (insn_tag[iter_var0] == insn_tag[iter_var1])) begin : dup
          err_dup_tag = 1'b1;
        end
      end
    end
  end

  // RT_TEMPORAL_PE_NO_MATCH: any input valid with tag matching no instruction
  always_comb begin : chk_no_match
    integer iter_var0;
    err_no_match = 1'b0;
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : per_in
      if (in_valid[iter_var0] && !in_match[iter_var0]) begin : no_match
        err_no_match = 1'b1;
      end
    end
  end

  // CFG_TEMPORAL_PE_ILLEGAL_REG and CFG_TEMPORAL_PE_REG_TAG_NONZERO:
  // Only elaborated when NUM_REGISTERS > 0 (generate-time guard) to avoid
  // negative-width part-selects from REG_BITS-1 when NUM_REGISTERS=0.
  generate
    if (NUM_REGISTERS > 0) begin : g_reg_checks
      always_comb begin : chk_illegal_reg
        integer iter_var0, iter_var1;
        err_illegal_reg = 1'b0;
        for (iter_var0 = 0; iter_var0 < NUM_INSTRUCTIONS; iter_var0 = iter_var0 + 1) begin : per_insn
          if (insn_valid[iter_var0]) begin : valid_insn
            automatic int insn_base = iter_var0 * INSN_WIDTH;
            for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1 = iter_var1 + 1) begin : per_op
              automatic int op_base = insn_base + NUM_OUTPUTS * RESULT_WIDTH + iter_var1 * REG_BITS;
              if (cfg_data[op_base + REG_BITS - 1]) begin : is_reg
                if ({{(32 - (REG_BITS - 1)){1'b0}}, cfg_data[op_base +: (REG_BITS - 1)]} >= 32'(NUM_REGISTERS)) begin : oob
                  err_illegal_reg = 1'b1;
                end
              end
            end
            for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1 = iter_var1 + 1) begin : per_res
              automatic int res_base = insn_base + iter_var1 * RESULT_WIDTH;
              if (cfg_data[res_base + RESULT_WIDTH - 1]) begin : is_reg
                if ({{(32 - (RES_BITS - 1)){1'b0}}, cfg_data[res_base + TAG_WIDTH +: (RES_BITS - 1)]} >= 32'(NUM_REGISTERS)) begin : oob
                  err_illegal_reg = 1'b1;
                end
              end
            end
          end
        end
      end

      always_comb begin : chk_reg_tag_nz
        integer iter_var0, iter_var1;
        err_reg_tag_nz = 1'b0;
        for (iter_var0 = 0; iter_var0 < NUM_INSTRUCTIONS; iter_var0 = iter_var0 + 1) begin : per_insn
          if (insn_valid[iter_var0]) begin : valid_insn
            automatic int insn_base = iter_var0 * INSN_WIDTH;
            for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1 = iter_var1 + 1) begin : per_res
              automatic int res_base = insn_base + iter_var1 * RESULT_WIDTH;
              if (cfg_data[res_base + RESULT_WIDTH - 1]) begin : is_reg
                if (cfg_data[res_base +: TAG_WIDTH] != '0) begin : nonzero_tag
                  err_reg_tag_nz = 1'b1;
                end
              end
            end
          end
        end
      end
    end else begin : g_no_reg_checks
      assign err_illegal_reg = 1'b0;
      assign err_reg_tag_nz  = 1'b0;
    end
  endgenerate

  // Priority-encode error code from individual flags
  logic        err_detect;
  logic [15:0] err_code_comb;

  always_comb begin : err_encode
    err_detect    = 1'b0;
    err_code_comb = 16'd0;

    if (err_dup_tag) begin : e_dup
      err_detect    = 1'b1;
      err_code_comb = CFG_TEMPORAL_PE_DUP_TAG;
    end else if (err_illegal_reg) begin : e_reg
      err_detect    = 1'b1;
      err_code_comb = CFG_TEMPORAL_PE_ILLEGAL_REG;
    end else if (err_reg_tag_nz) begin : e_rtag
      err_detect    = 1'b1;
      err_code_comb = CFG_TEMPORAL_PE_REG_TAG_NONZERO;
    end else if (err_no_match) begin : e_nomatch
      err_detect    = 1'b1;
      err_code_comb = RT_TEMPORAL_PE_NO_MATCH;
    end
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
