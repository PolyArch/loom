// fabric_temporal_pe_imem.sv -- Instruction memory and tag-match CAM.
//
// Stores NUM_INSTR instruction slots for the temporal PE.  Each slot
// is bit-packed low-to-high as defined in spec-fabric-config_mem.md:
//
//   valid | tag | opcode | operand_cfg[] | input_mux[] | output_demux[] | result_cfg[]
//
// Tag-matching: incoming tag is compared in parallel against all slots.
// The first valid slot whose tag matches wins (lowest index priority).
//
// Configuration words are loaded serially through cfg_valid / cfg_wdata.
// The module also unpacks the persistent per-FU config region that follows
// the instruction slots in the bitstream.

module fabric_temporal_pe_imem
  import fabric_pkg::*;
#(
  parameter int unsigned NUM_INSTR       = 4,
  parameter int unsigned NUM_FU          = 2,
  parameter int unsigned TAG_WIDTH       = 4,
  parameter int unsigned MAX_FU_IN       = 2,
  parameter int unsigned MAX_FU_OUT      = 2,
  parameter int unsigned NUM_REG         = 0,
  parameter int unsigned NUM_PE_IN       = 4,
  parameter int unsigned NUM_PE_OUT      = 4,
  parameter int unsigned DATA_WIDTH      = 32,
  // Total config bits for all FU persistent configs (sum of all per-FU bits).
  parameter int unsigned TOTAL_FU_CFG_BITS = 0
)(
  input  logic        clk,
  input  logic        rst_n,

  // --- Config loading (word-serial) ---
  input  logic        cfg_valid,
  input  logic [31:0] cfg_wdata,

  // --- Tag-match query ---
  input  logic                    query_valid,
  input  logic [TAG_WIDTH-1:0]    query_tag,

  // --- Match result ---
  output logic                    match_found,
  output logic [SLOT_IDX_W-1:0]  match_slot_idx,

  // --- Decoded slot fields for the matched slot (active only when match_found) ---
  output logic [OPCODE_W-1:0]    match_opcode,

  // Operand config: per MAX_FU_IN: {is_reg, reg_idx}
  output logic [MAX_FU_IN-1:0]   match_operand_is_reg,
  output logic [REG_IDX_W-1:0]   match_operand_reg_idx [MAX_FU_IN],

  // Input mux: per MAX_FU_IN: {disconnect, discard, sel}
  output logic [IN_SEL_W-1:0]    match_in_mux_sel      [MAX_FU_IN],
  output logic [MAX_FU_IN-1:0]   match_in_mux_discard,
  output logic [MAX_FU_IN-1:0]   match_in_mux_disconnect,

  // Output demux: per MAX_FU_OUT: {disconnect, discard, sel}
  output logic [OUT_SEL_W-1:0]   match_out_demux_sel      [MAX_FU_OUT],
  output logic [MAX_FU_OUT-1:0]  match_out_demux_discard,
  output logic [MAX_FU_OUT-1:0]  match_out_demux_disconnect,

  // Result config: per MAX_FU_OUT: {is_reg, reg_idx, result_tag}
  output logic [MAX_FU_OUT-1:0]  match_result_is_reg,
  output logic [REG_IDX_W-1:0]   match_result_reg_idx [MAX_FU_OUT],
  output logic [TAG_WIDTH-1:0]   match_result_tag     [MAX_FU_OUT],

  // --- Persistent FU config output ---
  output logic [FU_CFG_STORE_W-1:0] fu_cfg_bits
);

  // ---------------------------------------------------------------
  // Derived widths
  // ---------------------------------------------------------------
  localparam int unsigned SLOT_IDX_W = (NUM_INSTR > 1) ? $clog2(NUM_INSTR) : 1;
  localparam int unsigned OPCODE_W   = clog2(NUM_FU);
  localparam int unsigned REG_IDX_W  = clog2(NUM_REG);
  localparam int unsigned IN_SEL_W   = clog2(NUM_PE_IN);
  localparam int unsigned OUT_SEL_W  = clog2(NUM_PE_OUT);

  // Operand config width per operand: reg_idx + is_reg (only when NUM_REG > 0)
  localparam int unsigned OPERAND_CFG_W = (NUM_REG == 0) ? 0 : (REG_IDX_W + 1);

  // Input mux width per entry: sel + discard + disconnect
  localparam int unsigned IN_MUX_W = IN_SEL_W + 2;

  // Output demux width per entry: sel + discard + disconnect
  localparam int unsigned OUT_DEMUX_W = OUT_SEL_W + 2;

  // Result config width per entry: result_tag + (reg_idx + is_reg when regs exist)
  localparam int unsigned RESULT_CFG_W = TAG_WIDTH + OPERAND_CFG_W;

  // Per-slot total bit width
  localparam int unsigned SLOT_BITS = 1 + TAG_WIDTH + OPCODE_W
                                    + MAX_FU_IN  * OPERAND_CFG_W
                                    + MAX_FU_IN  * IN_MUX_W
                                    + MAX_FU_OUT * OUT_DEMUX_W
                                    + MAX_FU_OUT * RESULT_CFG_W;

  // Total config bits = all slots + persistent FU config
  localparam int unsigned TOTAL_CFG_BITS = NUM_INSTR * SLOT_BITS + TOTAL_FU_CFG_BITS;
  localparam int unsigned TOTAL_CFG_WORDS = (TOTAL_CFG_BITS + 31) / 32;

  // FU cfg storage width: at least 1 to avoid zero-width
  localparam int unsigned FU_CFG_STORE_W = (TOTAL_FU_CFG_BITS > 0) ? TOTAL_FU_CFG_BITS : 1;

  // Effective widths (at least 1 for port legality)
  localparam int unsigned EFF_OPCODE_W   = (OPCODE_W   > 0) ? OPCODE_W   : 1;
  localparam int unsigned EFF_REG_IDX_W  = (REG_IDX_W  > 0) ? REG_IDX_W  : 1;
  localparam int unsigned EFF_IN_SEL_W   = (IN_SEL_W   > 0) ? IN_SEL_W   : 1;
  localparam int unsigned EFF_OUT_SEL_W  = (OUT_SEL_W  > 0) ? OUT_SEL_W  : 1;

  // ---------------------------------------------------------------
  // Config storage -- flat bit vector built from word-serial load
  // ---------------------------------------------------------------
  logic [TOTAL_CFG_BITS-1:0] cfg_flat;
  logic [$clog2(TOTAL_CFG_WORDS > 1 ? TOTAL_CFG_WORDS : 2)-1:0] cfg_word_cnt;

  always_ff @(posedge clk or negedge rst_n) begin : cfg_load
    if (!rst_n) begin : cfg_load_reset
      cfg_flat     <= '0;
      cfg_word_cnt <= '0;
    end : cfg_load_reset
    else if (cfg_valid) begin : cfg_load_accept
      // Pack incoming word into flat vector at word_cnt position
      integer iter_var0;
      for (iter_var0 = 0; iter_var0 < 32; iter_var0 = iter_var0 + 1) begin : cfg_bit_pack
        if ((cfg_word_cnt * 32 + iter_var0) < TOTAL_CFG_BITS) begin : cfg_bit_in_range
          cfg_flat[cfg_word_cnt * 32 + iter_var0] <= cfg_wdata[iter_var0];
        end : cfg_bit_in_range
      end : cfg_bit_pack
      if (cfg_word_cnt == TOTAL_CFG_WORDS[$clog2(TOTAL_CFG_WORDS > 1 ? TOTAL_CFG_WORDS : 2)-1:0] - 1'b1) begin : cfg_wrap
        cfg_word_cnt <= '0;
      end : cfg_wrap
      else begin : cfg_advance
        cfg_word_cnt <= cfg_word_cnt + 1'b1;
      end : cfg_advance
    end : cfg_load_accept
  end : cfg_load

  // ---------------------------------------------------------------
  // Slot field extraction helpers
  // ---------------------------------------------------------------
  // Extract SLOT_BITS bits for slot [s] starting at bit s*SLOT_BITS
  // Then unpack fields in low-to-high order.

  // Per-slot decoded fields stored in arrays for CAM lookup
  logic                    slot_valid     [0:NUM_INSTR-1];
  logic [TAG_WIDTH-1:0]    slot_tag       [0:NUM_INSTR-1];
  logic [EFF_OPCODE_W-1:0] slot_opcode   [0:NUM_INSTR-1];

  // Operand config per slot
  logic [MAX_FU_IN-1:0]   slot_operand_is_reg [0:NUM_INSTR-1];
  logic [EFF_REG_IDX_W-1:0] slot_operand_reg_idx [0:NUM_INSTR-1][0:MAX_FU_IN-1];

  // Input mux per slot
  logic [EFF_IN_SEL_W-1:0] slot_in_mux_sel [0:NUM_INSTR-1][0:MAX_FU_IN-1];
  logic [MAX_FU_IN-1:0]   slot_in_mux_discard    [0:NUM_INSTR-1];
  logic [MAX_FU_IN-1:0]   slot_in_mux_disconnect  [0:NUM_INSTR-1];

  // Output demux per slot
  logic [EFF_OUT_SEL_W-1:0] slot_out_demux_sel [0:NUM_INSTR-1][0:MAX_FU_OUT-1];
  logic [MAX_FU_OUT-1:0]  slot_out_demux_discard    [0:NUM_INSTR-1];
  logic [MAX_FU_OUT-1:0]  slot_out_demux_disconnect  [0:NUM_INSTR-1];

  // Result config per slot
  logic [MAX_FU_OUT-1:0]  slot_result_is_reg   [0:NUM_INSTR-1];
  logic [EFF_REG_IDX_W-1:0] slot_result_reg_idx [0:NUM_INSTR-1][0:MAX_FU_OUT-1];
  logic [TAG_WIDTH-1:0]   slot_result_tag       [0:NUM_INSTR-1][0:MAX_FU_OUT-1];

  // ---------------------------------------------------------------
  // Combinational field extraction from cfg_flat
  // ---------------------------------------------------------------
  always_comb begin : slot_decode
    integer iter_var0;
    integer iter_var1;
    integer bit_pos;

    for (iter_var0 = 0; iter_var0 < NUM_INSTR; iter_var0 = iter_var0 + 1) begin : decode_slot
      bit_pos = iter_var0 * SLOT_BITS;

      // valid (1 bit)
      slot_valid[iter_var0] = cfg_flat[bit_pos];
      bit_pos = bit_pos + 1;

      // tag (TAG_WIDTH bits)
      slot_tag[iter_var0] = cfg_flat[bit_pos +: TAG_WIDTH];
      bit_pos = bit_pos + TAG_WIDTH;

      // opcode
      if (OPCODE_W > 0) begin : decode_opcode
        slot_opcode[iter_var0] = cfg_flat[bit_pos +: OPCODE_W];
      end : decode_opcode
      else begin : decode_opcode_zero
        slot_opcode[iter_var0] = '0;
      end : decode_opcode_zero
      bit_pos = bit_pos + OPCODE_W;

      // operand configs
      slot_operand_is_reg[iter_var0] = '0;
      for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : decode_operand
        slot_operand_reg_idx[iter_var0][iter_var1] = '0;
        if (OPERAND_CFG_W > 0) begin : has_operand_cfg
          if (REG_IDX_W > 0) begin : has_reg_idx
            slot_operand_reg_idx[iter_var0][iter_var1] = cfg_flat[bit_pos +: REG_IDX_W];
          end : has_reg_idx
          slot_operand_is_reg[iter_var0][iter_var1] = cfg_flat[bit_pos + REG_IDX_W];
          bit_pos = bit_pos + OPERAND_CFG_W;
        end : has_operand_cfg
      end : decode_operand

      // input mux controls
      slot_in_mux_discard[iter_var0]    = '0;
      slot_in_mux_disconnect[iter_var0] = '0;
      for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : decode_in_mux
        slot_in_mux_sel[iter_var0][iter_var1] = '0;
        if (IN_SEL_W > 0) begin : has_in_sel
          slot_in_mux_sel[iter_var0][iter_var1] = cfg_flat[bit_pos +: IN_SEL_W];
        end : has_in_sel
        slot_in_mux_discard[iter_var0][iter_var1]    = cfg_flat[bit_pos + IN_SEL_W];
        slot_in_mux_disconnect[iter_var0][iter_var1] = cfg_flat[bit_pos + IN_SEL_W + 1];
        bit_pos = bit_pos + IN_MUX_W;
      end : decode_in_mux

      // output demux controls
      slot_out_demux_discard[iter_var0]    = '0;
      slot_out_demux_disconnect[iter_var0] = '0;
      for (iter_var1 = 0; iter_var1 < MAX_FU_OUT; iter_var1 = iter_var1 + 1) begin : decode_out_demux
        slot_out_demux_sel[iter_var0][iter_var1] = '0;
        if (OUT_SEL_W > 0) begin : has_out_sel
          slot_out_demux_sel[iter_var0][iter_var1] = cfg_flat[bit_pos +: OUT_SEL_W];
        end : has_out_sel
        slot_out_demux_discard[iter_var0][iter_var1]    = cfg_flat[bit_pos + OUT_SEL_W];
        slot_out_demux_disconnect[iter_var0][iter_var1] = cfg_flat[bit_pos + OUT_SEL_W + 1];
        bit_pos = bit_pos + OUT_DEMUX_W;
      end : decode_out_demux

      // result configs
      slot_result_is_reg[iter_var0] = '0;
      for (iter_var1 = 0; iter_var1 < MAX_FU_OUT; iter_var1 = iter_var1 + 1) begin : decode_result
        // result_tag (TAG_WIDTH bits)
        slot_result_tag[iter_var0][iter_var1] = cfg_flat[bit_pos +: TAG_WIDTH];
        bit_pos = bit_pos + TAG_WIDTH;

        // reg_idx + is_reg (only when regs exist)
        slot_result_reg_idx[iter_var0][iter_var1] = '0;
        if (OPERAND_CFG_W > 0) begin : has_result_reg
          if (REG_IDX_W > 0) begin : has_result_reg_idx
            slot_result_reg_idx[iter_var0][iter_var1] = cfg_flat[bit_pos +: REG_IDX_W];
          end : has_result_reg_idx
          slot_result_is_reg[iter_var0][iter_var1] = cfg_flat[bit_pos + REG_IDX_W];
          bit_pos = bit_pos + OPERAND_CFG_W;
        end : has_result_reg
      end : decode_result
    end : decode_slot
  end : slot_decode

  // ---------------------------------------------------------------
  // Persistent FU config extraction
  // ---------------------------------------------------------------
  generate
    if (TOTAL_FU_CFG_BITS > 0) begin : gen_fu_cfg
      assign fu_cfg_bits = cfg_flat[NUM_INSTR * SLOT_BITS +: TOTAL_FU_CFG_BITS];
    end : gen_fu_cfg
    else begin : gen_no_fu_cfg
      assign fu_cfg_bits = '0;
    end : gen_no_fu_cfg
  endgenerate

  // ---------------------------------------------------------------
  // Tag-match CAM: first valid slot whose tag matches query_tag
  // ---------------------------------------------------------------
  always_comb begin : tag_match
    integer iter_var0;

    match_found    = 1'b0;
    match_slot_idx = '0;
    match_opcode   = '0;

    match_operand_is_reg     = '0;
    match_in_mux_discard     = '0;
    match_in_mux_disconnect  = '0;
    match_out_demux_discard    = '0;
    match_out_demux_disconnect = '0;
    match_result_is_reg      = '0;

    for (iter_var0 = 0; iter_var0 < MAX_FU_IN; iter_var0 = iter_var0 + 1) begin : def_in
      match_operand_reg_idx[iter_var0] = '0;
      match_in_mux_sel[iter_var0]      = '0;
    end : def_in

    for (iter_var0 = 0; iter_var0 < MAX_FU_OUT; iter_var0 = iter_var0 + 1) begin : def_out
      match_out_demux_sel[iter_var0] = '0;
      match_result_reg_idx[iter_var0] = '0;
      match_result_tag[iter_var0]     = '0;
    end : def_out

    if (query_valid) begin : query_active
      // Reverse scan so lowest matching index overwrites
      for (iter_var0 = NUM_INSTR - 1; iter_var0 >= 0; iter_var0 = iter_var0 - 1) begin : slot_scan
        if (slot_valid[iter_var0] && (slot_tag[iter_var0] == query_tag)) begin : slot_hit
          match_found    = 1'b1;
          match_slot_idx = iter_var0[SLOT_IDX_W-1:0];
          match_opcode   = slot_opcode[iter_var0];

          match_operand_is_reg     = slot_operand_is_reg[iter_var0];
          match_in_mux_discard     = slot_in_mux_discard[iter_var0];
          match_in_mux_disconnect  = slot_in_mux_disconnect[iter_var0];
          match_out_demux_discard    = slot_out_demux_discard[iter_var0];
          match_out_demux_disconnect = slot_out_demux_disconnect[iter_var0];
          match_result_is_reg      = slot_result_is_reg[iter_var0];

          // Copy array fields using inner variable
          // (SystemVerilog allows this in combinational blocks)
          begin : copy_arrays
            integer iter_var1;
            for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : copy_in
              match_operand_reg_idx[iter_var1] = slot_operand_reg_idx[iter_var0][iter_var1];
              match_in_mux_sel[iter_var1]      = slot_in_mux_sel[iter_var0][iter_var1];
            end : copy_in
            for (iter_var1 = 0; iter_var1 < MAX_FU_OUT; iter_var1 = iter_var1 + 1) begin : copy_out
              match_out_demux_sel[iter_var1]  = slot_out_demux_sel[iter_var0][iter_var1];
              match_result_reg_idx[iter_var1] = slot_result_reg_idx[iter_var0][iter_var1];
              match_result_tag[iter_var1]     = slot_result_tag[iter_var0][iter_var1];
            end : copy_out
          end : copy_arrays
        end : slot_hit
      end : slot_scan
    end : query_active
  end : tag_match

endmodule : fabric_temporal_pe_imem
