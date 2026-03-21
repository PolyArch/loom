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
//
// All decoded slot fields are exposed as output arrays so the top-level PE
// can use them for operand routing, scheduling, and output arbitration
// without re-decoding the bitstream.

module fabric_temporal_pe_imem
  import fabric_pkg::*;
#(
  parameter int unsigned NUM_INSTR         = 4,
  parameter int unsigned NUM_FU            = 2,
  parameter int unsigned TAG_WIDTH         = 4,
  parameter int unsigned MAX_FU_IN         = 2,
  parameter int unsigned MAX_FU_OUT        = 2,
  parameter int unsigned NUM_REG           = 0,
  parameter int unsigned NUM_PE_IN         = 4,
  parameter int unsigned NUM_PE_OUT        = 4,
  parameter int unsigned DATA_WIDTH        = 32,
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
  output logic                    match_found,
  output logic [(NUM_INSTR > 1 ? $clog2(NUM_INSTR) : 1)-1:0] match_slot_idx,

  // --- All slot fields (exposed for top-level use) ---
  output logic                    slot_valid          [0:NUM_INSTR-1],
  output logic [TAG_WIDTH-1:0]    slot_tag            [0:NUM_INSTR-1],
  output logic [(NUM_FU > 1 ? $clog2(NUM_FU) : 1)-1:0]
                                  slot_opcode         [0:NUM_INSTR-1],

  output logic [MAX_FU_IN-1:0]   slot_operand_is_reg [0:NUM_INSTR-1],
  output logic [(NUM_REG > 1 ? $clog2(NUM_REG) : 1)-1:0]
                                  slot_operand_reg_idx[0:NUM_INSTR-1][0:MAX_FU_IN-1],

  output logic [(NUM_PE_IN > 1 ? $clog2(NUM_PE_IN) : 1)-1:0]
                                  slot_in_mux_sel     [0:NUM_INSTR-1][0:MAX_FU_IN-1],
  output logic [MAX_FU_IN-1:0]   slot_in_mux_discard   [0:NUM_INSTR-1],
  output logic [MAX_FU_IN-1:0]   slot_in_mux_disconnect[0:NUM_INSTR-1],

  output logic [(NUM_PE_OUT > 1 ? $clog2(NUM_PE_OUT) : 1)-1:0]
                                  slot_out_demux_sel     [0:NUM_INSTR-1][0:MAX_FU_OUT-1],
  output logic [MAX_FU_OUT-1:0]  slot_out_demux_discard   [0:NUM_INSTR-1],
  output logic [MAX_FU_OUT-1:0]  slot_out_demux_disconnect[0:NUM_INSTR-1],

  output logic [MAX_FU_OUT-1:0]  slot_result_is_reg   [0:NUM_INSTR-1],
  output logic [(NUM_REG > 1 ? $clog2(NUM_REG) : 1)-1:0]
                                  slot_result_reg_idx  [0:NUM_INSTR-1][0:MAX_FU_OUT-1],
  output logic [TAG_WIDTH-1:0]   slot_result_tag      [0:NUM_INSTR-1][0:MAX_FU_OUT-1],

  // --- Persistent FU config output ---
  output logic [(TOTAL_FU_CFG_BITS > 0 ? TOTAL_FU_CFG_BITS : 1)-1:0] fu_cfg_bits
);

  // ---------------------------------------------------------------
  // Derived widths
  // ---------------------------------------------------------------
  localparam int unsigned SLOT_IDX_W  = (NUM_INSTR > 1) ? $clog2(NUM_INSTR) : 1;
  localparam int unsigned OPCODE_W    = clog2(NUM_FU);
  localparam int unsigned REG_IDX_W   = clog2(NUM_REG);
  localparam int unsigned IN_SEL_W    = clog2(NUM_PE_IN);
  localparam int unsigned OUT_SEL_W   = clog2(NUM_PE_OUT);
  localparam int unsigned EFF_OPCODE  = (OPCODE_W  > 0) ? OPCODE_W  : 1;
  localparam int unsigned EFF_REG_IDX = (REG_IDX_W > 0) ? REG_IDX_W : 1;
  localparam int unsigned EFF_IN_SEL  = (IN_SEL_W  > 0) ? IN_SEL_W  : 1;
  localparam int unsigned EFF_OUT_SEL = (OUT_SEL_W > 0) ? OUT_SEL_W : 1;

  localparam int unsigned OPERAND_CFG_W = (NUM_REG == 0) ? 0 : (REG_IDX_W + 1);
  localparam int unsigned IN_MUX_W      = IN_SEL_W + 2;
  localparam int unsigned OUT_DEMUX_W   = OUT_SEL_W + 2;
  localparam int unsigned RESULT_CFG_W  = TAG_WIDTH + OPERAND_CFG_W;

  localparam int unsigned SLOT_BITS = 1 + TAG_WIDTH + OPCODE_W
                                    + MAX_FU_IN  * OPERAND_CFG_W
                                    + MAX_FU_IN  * IN_MUX_W
                                    + MAX_FU_OUT * OUT_DEMUX_W
                                    + MAX_FU_OUT * RESULT_CFG_W;

  localparam int unsigned TOTAL_CFG_BITS  = NUM_INSTR * SLOT_BITS + TOTAL_FU_CFG_BITS;
  localparam int unsigned TOTAL_CFG_WORDS = (TOTAL_CFG_BITS + 31) / 32;
  localparam int unsigned CFG_CNT_W = (TOTAL_CFG_WORDS > 1) ? $clog2(TOTAL_CFG_WORDS) : 1;

  // ---------------------------------------------------------------
  // Config storage -- flat bit vector built from word-serial load
  // ---------------------------------------------------------------
  logic [TOTAL_CFG_BITS-1:0] cfg_flat;
  logic [CFG_CNT_W-1:0]     cfg_word_cnt;

  always_ff @(posedge clk or negedge rst_n) begin : cfg_load
    if (!rst_n) begin : cfg_load_reset
      cfg_flat     <= '0;
      cfg_word_cnt <= '0;
    end : cfg_load_reset
    else if (cfg_valid) begin : cfg_load_accept
      integer iter_var0;
      for (iter_var0 = 0; iter_var0 < 32; iter_var0 = iter_var0 + 1) begin : cfg_bit_pack
        if ((int'(cfg_word_cnt) * 32 + iter_var0) < TOTAL_CFG_BITS) begin : cfg_bit_in_range
          cfg_flat[int'(cfg_word_cnt) * 32 + iter_var0] <= cfg_wdata[iter_var0];
        end : cfg_bit_in_range
      end : cfg_bit_pack
      if (cfg_word_cnt == CFG_CNT_W'(TOTAL_CFG_WORDS - 1)) begin : cfg_wrap
        cfg_word_cnt <= '0;
      end : cfg_wrap
      else begin : cfg_advance
        cfg_word_cnt <= cfg_word_cnt + 1'b1;
      end : cfg_advance
    end : cfg_load_accept
  end : cfg_load

  // ---------------------------------------------------------------
  // Combinational field extraction from cfg_flat into output arrays.
  //
  // Bit-extraction widths must be >= 1 for +: operator legality.
  // We use EFF_* widths (min 1) and guard assignment with generate-time
  // conditions where the field truly has zero bits.
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

      // opcode (EFF_OPCODE bits, always >= 1)
      slot_opcode[iter_var0] = cfg_flat[bit_pos +: EFF_OPCODE];
      bit_pos = bit_pos + OPCODE_W;

      // operand configs
      slot_operand_is_reg[iter_var0] = '0;
      for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : decode_operand
        slot_operand_reg_idx[iter_var0][iter_var1] = '0;
      end : decode_operand
      // Only decode operand config fields if registers exist
      if (OPERAND_CFG_W > 0) begin : has_operand_cfgs
        for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : decode_operand_cfg
          slot_operand_reg_idx[iter_var0][iter_var1] = cfg_flat[bit_pos +: EFF_REG_IDX];
          slot_operand_is_reg[iter_var0][iter_var1]  = cfg_flat[bit_pos + REG_IDX_W];
          bit_pos = bit_pos + OPERAND_CFG_W;
        end : decode_operand_cfg
      end : has_operand_cfgs

      // input mux controls (always at least 2 bits: discard + disconnect)
      slot_in_mux_discard[iter_var0]    = '0;
      slot_in_mux_disconnect[iter_var0] = '0;
      for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : decode_in_mux
        slot_in_mux_sel[iter_var0][iter_var1] = '0;
        if (IN_SEL_W > 0) begin : has_in_sel
          slot_in_mux_sel[iter_var0][iter_var1] = cfg_flat[bit_pos +: EFF_IN_SEL];
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
          slot_out_demux_sel[iter_var0][iter_var1] = cfg_flat[bit_pos +: EFF_OUT_SEL];
        end : has_out_sel
        slot_out_demux_discard[iter_var0][iter_var1]    = cfg_flat[bit_pos + OUT_SEL_W];
        slot_out_demux_disconnect[iter_var0][iter_var1] = cfg_flat[bit_pos + OUT_SEL_W + 1];
        bit_pos = bit_pos + OUT_DEMUX_W;
      end : decode_out_demux

      // result configs
      slot_result_is_reg[iter_var0] = '0;
      for (iter_var1 = 0; iter_var1 < MAX_FU_OUT; iter_var1 = iter_var1 + 1) begin : decode_result
        slot_result_tag[iter_var0][iter_var1] = cfg_flat[bit_pos +: TAG_WIDTH];
        bit_pos = bit_pos + TAG_WIDTH;
        slot_result_reg_idx[iter_var0][iter_var1] = '0;
        if (OPERAND_CFG_W > 0) begin : has_result_reg
          slot_result_reg_idx[iter_var0][iter_var1] = cfg_flat[bit_pos +: EFF_REG_IDX];
          slot_result_is_reg[iter_var0][iter_var1]  = cfg_flat[bit_pos + REG_IDX_W];
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

    if (query_valid) begin : query_active
      // Reverse scan so lowest matching index overwrites
      for (iter_var0 = NUM_INSTR - 1; iter_var0 >= 0; iter_var0 = iter_var0 - 1) begin : slot_scan
        if (slot_valid[iter_var0] && (slot_tag[iter_var0] == query_tag)) begin : slot_hit
          match_found    = 1'b1;
          match_slot_idx = iter_var0[SLOT_IDX_W-1:0];
        end : slot_hit
      end : slot_scan
    end : query_active
  end : tag_match

endmodule : fabric_temporal_pe_imem
