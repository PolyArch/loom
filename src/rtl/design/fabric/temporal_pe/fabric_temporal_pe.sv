// fabric_temporal_pe.sv -- Top-level temporal PE container.
//
// Wires together:
//   - fabric_temporal_pe_imem      (instruction memory + tag-match CAM)
//   - fabric_temporal_pe_operand   (operand routing and buffering)
//   - fabric_temporal_pe_regfile   (FIFO-based register file)
//   - fabric_temporal_pe_scheduler (at-most-one FU fires per cycle)
//   - fabric_temporal_pe_fu_slot   (per-FU latency pipeline + interval throttle)
//   - fabric_temporal_pe_output_arb(FU-local output regs + round-robin arbitration)
//
// The actual FU body modules are NOT instantiated here -- they are
// connected externally through the fu_in_*/fu_out_* ports.  The C++
// SVGen library generates a PE wrapper that instantiates specific FU
// bodies and connects them to these ports.
//
// Config loading: word-serial through cfg_valid / cfg_wdata.  The
// bitstream layout follows spec-fabric-config_mem.md: instruction slots
// (low-to-high) followed by persistent per-FU config.

module fabric_temporal_pe
  import fabric_pkg::*;
#(
  parameter int unsigned NUM_IN       = 4,
  parameter int unsigned NUM_OUT      = 4,
  parameter int unsigned DATA_WIDTH   = 32,
  parameter int unsigned TAG_WIDTH    = 4,
  parameter int unsigned NUM_FU       = 2,
  parameter int unsigned NUM_INSTR    = 4,
  parameter int unsigned NUM_REG      = 0,
  parameter int unsigned REG_FIFO_DEPTH = 4,
  parameter int unsigned MAX_FU_IN    = 2,
  parameter int unsigned MAX_FU_OUT   = 2,
  parameter bit          ENABLE_SHARE_OPERAND_BUF = 1'b0,
  parameter int unsigned OPERAND_BUF_SIZE = 8,
  // Total persistent FU config bits (sum across all FUs).
  parameter int unsigned TOTAL_FU_CFG_BITS = 0,
  // Per-FU parameters passed as packed arrays.
  // fu_num_inputs[f]: number of inputs FU f has.
  // fu_num_outputs[f]: number of outputs FU f has.
  // fu_latency[f]: configured latency for FU f (-1 for dataflow).
  // fu_intrinsic_lat[f]: intrinsic latency of FU f body.
  // fu_interval[f]: configured interval for FU f (-1 for dataflow).
  // These are exposed as flat arrays; the top-level wrapper sets them.
  parameter int unsigned FU_NUM_INPUTS  [NUM_FU] = '{default: MAX_FU_IN},
  parameter int unsigned FU_NUM_OUTPUTS [NUM_FU] = '{default: MAX_FU_OUT},
  parameter int signed   FU_LATENCY     [NUM_FU] = '{default: 0},
  parameter int unsigned FU_INTRINSIC   [NUM_FU] = '{default: 0},
  parameter int signed   FU_INTERVAL    [NUM_FU] = '{default: 1}
)(
  input  logic        clk,
  input  logic        rst_n,

  // --- Config loading ---
  input  logic        cfg_valid,
  input  logic [31:0] cfg_wdata,

  // --- PE input ports (handshake: valid/ready/data/tag) ---
  input  logic [NUM_IN-1:0]       pe_in_valid,
  output logic [NUM_IN-1:0]       pe_in_ready,
  input  logic [DATA_WIDTH-1:0]   pe_in_data  [NUM_IN],
  input  logic [TAG_WIDTH-1:0]    pe_in_tag   [NUM_IN],

  // --- PE output ports (handshake: valid/ready/data/tag) ---
  output logic [NUM_OUT-1:0]      pe_out_valid,
  input  logic [NUM_OUT-1:0]      pe_out_ready,
  output logic [DATA_WIDTH-1:0]   pe_out_data [NUM_OUT],
  output logic [TAG_WIDTH-1:0]    pe_out_tag  [NUM_OUT],

  // --- FU body interface (external FU instances connect here) ---
  // Per-FU input: driven by this PE toward FU body
  output logic                     fu_fire     [NUM_FU],
  output logic [DATA_WIDTH-1:0]   fu_in_data  [NUM_FU][MAX_FU_IN],
  output logic [MAX_FU_IN-1:0]    fu_in_valid [NUM_FU],

  // Per-FU output: driven by FU body toward this PE
  input  logic [MAX_FU_OUT-1:0]   fu_out_valid [NUM_FU],
  input  logic [DATA_WIDTH-1:0]   fu_out_data  [NUM_FU][MAX_FU_OUT],

  // --- Persistent FU config bits (to FU bodies) ---
  // Width: max(1, TOTAL_FU_CFG_BITS) to avoid zero-width ports
  output logic [(TOTAL_FU_CFG_BITS > 0 ? TOTAL_FU_CFG_BITS : 1)-1:0] fu_cfg_bits
);

  // ---------------------------------------------------------------
  // Derived parameters
  // ---------------------------------------------------------------
  localparam int unsigned SLOT_IDX_W  = (NUM_INSTR > 1) ? $clog2(NUM_INSTR) : 1;
  localparam int unsigned OPCODE_W    = clog2(NUM_FU);
  localparam int unsigned FU_IDX_W    = (NUM_FU > 1) ? $clog2(NUM_FU) : 1;
  localparam int unsigned REG_IDX_W   = clog2(NUM_REG);
  localparam int unsigned IN_SEL_W    = clog2(NUM_IN);
  localparam int unsigned OUT_SEL_W   = clog2(NUM_OUT);
  localparam int unsigned FU_IN_CNT_W = $clog2(MAX_FU_IN + 1);
  localparam int unsigned FU_CFG_W    = (TOTAL_FU_CFG_BITS > 0) ? TOTAL_FU_CFG_BITS : 1;

  localparam int unsigned EFF_OPCODE  = (OPCODE_W  > 0) ? OPCODE_W  : 1;
  localparam int unsigned EFF_REG_IDX = (REG_IDX_W > 0) ? REG_IDX_W : 1;
  localparam int unsigned EFF_IN_SEL  = (IN_SEL_W  > 0) ? IN_SEL_W  : 1;
  localparam int unsigned EFF_OUT_SEL = (OUT_SEL_W > 0) ? OUT_SEL_W : 1;

  // ---------------------------------------------------------------
  // Internal wires: IMEM outputs
  // ---------------------------------------------------------------
  logic                    imem_match_found;
  logic [SLOT_IDX_W-1:0]  imem_match_slot;
  logic [EFF_OPCODE-1:0]  imem_match_opcode;
  logic [MAX_FU_IN-1:0]   imem_match_operand_is_reg;
  logic [EFF_REG_IDX-1:0] imem_match_operand_reg_idx [MAX_FU_IN];
  logic [EFF_IN_SEL-1:0]  imem_match_in_mux_sel      [MAX_FU_IN];
  logic [MAX_FU_IN-1:0]   imem_match_in_mux_discard;
  logic [MAX_FU_IN-1:0]   imem_match_in_mux_disconnect;
  logic [EFF_OUT_SEL-1:0] imem_match_out_demux_sel      [MAX_FU_OUT];
  logic [MAX_FU_OUT-1:0]  imem_match_out_demux_discard;
  logic [MAX_FU_OUT-1:0]  imem_match_out_demux_disconnect;
  logic [MAX_FU_OUT-1:0]  imem_match_result_is_reg;
  logic [EFF_REG_IDX-1:0] imem_match_result_reg_idx [MAX_FU_OUT];
  logic [TAG_WIDTH-1:0]   imem_match_result_tag     [MAX_FU_OUT];

  // We also need per-slot decoded fields for the operand module and scheduler.
  // The imem stores these internally; we re-extract them from cfg_flat.
  // For simplicity, we keep the imem as the config storage and expose
  // all slot fields as direct outputs from the imem's internal arrays.
  // However, since the imem only exposes the MATCHED slot, we need
  // a parallel path for all slots' fields for the operand module.

  // To keep the design clean, we share the cfg_flat from imem and
  // decode slot fields both inside imem and here at the top level
  // for the operand and scheduler connections.

  // ---------------------------------------------------------------
  // IMEM instantiation
  // ---------------------------------------------------------------
  fabric_temporal_pe_imem #(
    .NUM_INSTR        (NUM_INSTR),
    .NUM_FU           (NUM_FU),
    .TAG_WIDTH        (TAG_WIDTH),
    .MAX_FU_IN        (MAX_FU_IN),
    .MAX_FU_OUT       (MAX_FU_OUT),
    .NUM_REG          (NUM_REG),
    .NUM_PE_IN        (NUM_IN),
    .NUM_PE_OUT       (NUM_OUT),
    .DATA_WIDTH       (DATA_WIDTH),
    .TOTAL_FU_CFG_BITS(TOTAL_FU_CFG_BITS)
  ) u_imem (
    .clk       (clk),
    .rst_n     (rst_n),
    .cfg_valid (cfg_valid),
    .cfg_wdata (cfg_wdata),

    // Query: for now, we run a match for scheduler purposes.
    // The scheduler scans all slots directly, but we can use
    // the imem match for ingress tag-matching.
    .query_valid (1'b0),
    .query_tag   ('0),

    .match_found    (imem_match_found),
    .match_slot_idx (imem_match_slot),
    .match_opcode   (imem_match_opcode),

    .match_operand_is_reg     (imem_match_operand_is_reg),
    .match_operand_reg_idx    (imem_match_operand_reg_idx),
    .match_in_mux_sel         (imem_match_in_mux_sel),
    .match_in_mux_discard     (imem_match_in_mux_discard),
    .match_in_mux_disconnect  (imem_match_in_mux_disconnect),
    .match_out_demux_sel         (imem_match_out_demux_sel),
    .match_out_demux_discard     (imem_match_out_demux_discard),
    .match_out_demux_disconnect  (imem_match_out_demux_disconnect),
    .match_result_is_reg      (imem_match_result_is_reg),
    .match_result_reg_idx     (imem_match_result_reg_idx),
    .match_result_tag         (imem_match_result_tag),

    .fu_cfg_bits (fu_cfg_bits)
  );

  // ---------------------------------------------------------------
  // Access to all slot fields for operand/scheduler
  //
  // We access the imem's internal slot arrays.  In a real hierarchical
  // design, these would be exposed as additional imem output ports.
  // For synthesizability, we replicate the decode logic here using
  // the same cfg_flat from the imem.  We use a generate-based approach:
  // the imem's cfg_flat is not directly accessible, so we maintain a
  // shadow copy in this top module.
  // ---------------------------------------------------------------
  localparam int unsigned OPERAND_CFG_W = (NUM_REG == 0) ? 0 : (REG_IDX_W + 1);
  localparam int unsigned IN_MUX_W      = IN_SEL_W + 2;
  localparam int unsigned OUT_DEMUX_W   = OUT_SEL_W + 2;
  localparam int unsigned RESULT_CFG_W  = TAG_WIDTH + OPERAND_CFG_W;
  localparam int unsigned SLOT_BITS     = 1 + TAG_WIDTH + OPCODE_W
                                        + MAX_FU_IN  * OPERAND_CFG_W
                                        + MAX_FU_IN  * IN_MUX_W
                                        + MAX_FU_OUT * OUT_DEMUX_W
                                        + MAX_FU_OUT * RESULT_CFG_W;
  localparam int unsigned TOTAL_CFG_BITS = NUM_INSTR * SLOT_BITS + TOTAL_FU_CFG_BITS;
  localparam int unsigned TOTAL_CFG_WORDS = (TOTAL_CFG_BITS + 31) / 32;

  // Shadow config flat register (same as imem's)
  logic [TOTAL_CFG_BITS-1:0] cfg_flat;
  logic [$clog2(TOTAL_CFG_WORDS > 1 ? TOTAL_CFG_WORDS : 2)-1:0] cfg_word_cnt;

  always_ff @(posedge clk or negedge rst_n) begin : shadow_cfg_load
    if (!rst_n) begin : shadow_reset
      cfg_flat     <= '0;
      cfg_word_cnt <= '0;
    end : shadow_reset
    else if (cfg_valid) begin : shadow_accept
      integer iter_var0;
      for (iter_var0 = 0; iter_var0 < 32; iter_var0 = iter_var0 + 1) begin : shadow_bit
        if ((cfg_word_cnt * 32 + iter_var0) < TOTAL_CFG_BITS) begin : shadow_in_range
          cfg_flat[cfg_word_cnt * 32 + iter_var0] <= cfg_wdata[iter_var0];
        end : shadow_in_range
      end : shadow_bit
      if (cfg_word_cnt == TOTAL_CFG_WORDS[$clog2(TOTAL_CFG_WORDS > 1 ? TOTAL_CFG_WORDS : 2)-1:0] - 1'b1) begin : shadow_wrap
        cfg_word_cnt <= '0;
      end : shadow_wrap
      else begin : shadow_inc
        cfg_word_cnt <= cfg_word_cnt + 1'b1;
      end : shadow_inc
    end : shadow_accept
  end : shadow_cfg_load

  // Decode all slot fields from shadow cfg_flat
  logic                     all_slot_valid              [0:NUM_INSTR-1];
  logic [TAG_WIDTH-1:0]     all_slot_tag                [0:NUM_INSTR-1];
  logic [EFF_OPCODE-1:0]    all_slot_opcode             [0:NUM_INSTR-1];
  logic                      all_slot_operand_is_reg    [0:NUM_INSTR-1][0:MAX_FU_IN-1];
  logic [EFF_REG_IDX-1:0]   all_slot_operand_reg_idx   [0:NUM_INSTR-1][0:MAX_FU_IN-1];
  logic [EFF_IN_SEL-1:0]    all_slot_in_mux_sel        [0:NUM_INSTR-1][0:MAX_FU_IN-1];
  logic                      all_slot_in_mux_discard    [0:NUM_INSTR-1][0:MAX_FU_IN-1];
  logic                      all_slot_in_mux_disconnect [0:NUM_INSTR-1][0:MAX_FU_IN-1];
  logic [EFF_OUT_SEL-1:0]   all_slot_out_demux_sel        [0:NUM_INSTR-1][0:MAX_FU_OUT-1];
  logic                      all_slot_out_demux_discard    [0:NUM_INSTR-1][0:MAX_FU_OUT-1];
  logic                      all_slot_out_demux_disconnect [0:NUM_INSTR-1][0:MAX_FU_OUT-1];
  logic [TAG_WIDTH-1:0]     all_slot_result_tag            [0:NUM_INSTR-1][0:MAX_FU_OUT-1];
  logic                      all_slot_result_is_reg        [0:NUM_INSTR-1][0:MAX_FU_OUT-1];
  logic [EFF_REG_IDX-1:0]   all_slot_result_reg_idx       [0:NUM_INSTR-1][0:MAX_FU_OUT-1];

  always_comb begin : all_slot_decode
    integer iter_var0;
    integer iter_var1;
    integer bit_pos;

    for (iter_var0 = 0; iter_var0 < NUM_INSTR; iter_var0 = iter_var0 + 1) begin : dec_slot
      bit_pos = iter_var0 * SLOT_BITS;

      all_slot_valid[iter_var0] = cfg_flat[bit_pos];
      bit_pos = bit_pos + 1;

      all_slot_tag[iter_var0] = cfg_flat[bit_pos +: TAG_WIDTH];
      bit_pos = bit_pos + TAG_WIDTH;

      if (OPCODE_W > 0) begin : dec_opc
        all_slot_opcode[iter_var0] = cfg_flat[bit_pos +: OPCODE_W];
      end : dec_opc
      else begin : dec_opc_z
        all_slot_opcode[iter_var0] = '0;
      end : dec_opc_z
      bit_pos = bit_pos + OPCODE_W;

      for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : dec_operand
        all_slot_operand_is_reg[iter_var0][iter_var1] = 1'b0;
        all_slot_operand_reg_idx[iter_var0][iter_var1] = '0;
        if (OPERAND_CFG_W > 0) begin : has_ocfg
          if (REG_IDX_W > 0) begin : has_ridx
            all_slot_operand_reg_idx[iter_var0][iter_var1] = cfg_flat[bit_pos +: REG_IDX_W];
          end : has_ridx
          all_slot_operand_is_reg[iter_var0][iter_var1] = cfg_flat[bit_pos + REG_IDX_W];
          bit_pos = bit_pos + OPERAND_CFG_W;
        end : has_ocfg
      end : dec_operand

      for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : dec_in_mux
        all_slot_in_mux_sel[iter_var0][iter_var1] = '0;
        if (IN_SEL_W > 0) begin : has_isel
          all_slot_in_mux_sel[iter_var0][iter_var1] = cfg_flat[bit_pos +: IN_SEL_W];
        end : has_isel
        all_slot_in_mux_discard[iter_var0][iter_var1] = cfg_flat[bit_pos + IN_SEL_W];
        all_slot_in_mux_disconnect[iter_var0][iter_var1] = cfg_flat[bit_pos + IN_SEL_W + 1];
        bit_pos = bit_pos + IN_MUX_W;
      end : dec_in_mux

      for (iter_var1 = 0; iter_var1 < MAX_FU_OUT; iter_var1 = iter_var1 + 1) begin : dec_out_demux
        all_slot_out_demux_sel[iter_var0][iter_var1] = '0;
        if (OUT_SEL_W > 0) begin : has_osel
          all_slot_out_demux_sel[iter_var0][iter_var1] = cfg_flat[bit_pos +: OUT_SEL_W];
        end : has_osel
        all_slot_out_demux_discard[iter_var0][iter_var1] = cfg_flat[bit_pos + OUT_SEL_W];
        all_slot_out_demux_disconnect[iter_var0][iter_var1] = cfg_flat[bit_pos + OUT_SEL_W + 1];
        bit_pos = bit_pos + OUT_DEMUX_W;
      end : dec_out_demux

      for (iter_var1 = 0; iter_var1 < MAX_FU_OUT; iter_var1 = iter_var1 + 1) begin : dec_result
        all_slot_result_tag[iter_var0][iter_var1] = cfg_flat[bit_pos +: TAG_WIDTH];
        bit_pos = bit_pos + TAG_WIDTH;
        all_slot_result_is_reg[iter_var0][iter_var1] = 1'b0;
        all_slot_result_reg_idx[iter_var0][iter_var1] = '0;
        if (OPERAND_CFG_W > 0) begin : has_rcfg
          if (REG_IDX_W > 0) begin : has_rridx
            all_slot_result_reg_idx[iter_var0][iter_var1] = cfg_flat[bit_pos +: REG_IDX_W];
          end : has_rridx
          all_slot_result_is_reg[iter_var0][iter_var1] = cfg_flat[bit_pos + REG_IDX_W];
          bit_pos = bit_pos + OPERAND_CFG_W;
        end : has_rcfg
      end : dec_result
    end : dec_slot
  end : all_slot_decode

  // ---------------------------------------------------------------
  // Operand module
  // ---------------------------------------------------------------
  logic [MAX_FU_IN-1:0]   sched_operand_ready [0:NUM_INSTR-1];
  logic [DATA_WIDTH-1:0]  sched_operand_data  [0:NUM_INSTR-1][0:MAX_FU_IN-1];

  // We need per-slot operand readiness for the scheduler.
  // The operand module supports one query at a time. We query each
  // slot combinationally in a generate loop using separate instances,
  // OR we use a single operand module and query per-slot in the
  // scheduler's slot scan.
  //
  // For area efficiency, we use ONE operand module and expose its
  // internal buffer state.  The scheduler checks readiness per slot.
  //
  // Approach: instantiate one operand module that handles all ingress
  // capture.  For readiness, we directly peek into the module's
  // internal state.  Since SV does not allow peeking into submodule
  // internals, we add per-slot query outputs or use a multi-query interface.
  //
  // Practical approach: we do NOT use the operand module's query port
  // for scheduling.  Instead, we replicate the readiness check in
  // the scheduler using the same buffer state.  To enable this, we
  // expose the buffer valid flags from the operand module.

  // For per-instruction mode: we need buf_valid[slot][operand]
  // For shared mode: we need FIFO head entry's per-operand valid

  // Since the operand module's internal state is not easily exposed
  // without adding more output ports, we restructure: the operand
  // module will be connected for ingress and consume only.  The
  // scheduler queries operand readiness through dedicated per-slot
  // query wires driven combinationally from the operand module.

  // The fire_slot_idx from the scheduler tells us which slot to query
  // for the actual FU input data.  The scheduler itself needs to check
  // ALL slots for readiness, so we iterate.

  // Simple approach: instantiate the operand module, then use a
  // multiplexed query loop.  For the scheduler, we check readiness
  // by iterating queries.

  // To keep synthesis clean, we directly maintain buffer state in this
  // top module for readiness checks, and use the operand module for
  // ingress/consume/data output.

  // REVISED: Use per-slot operand readiness arrays exposed from a
  // simplified buffer structure managed at this level.

  // ---------------------------------------------------------------
  // Per-instruction operand buffers (managed here for visibility)
  // ---------------------------------------------------------------
  logic                     obuf_valid [0:NUM_INSTR-1][0:MAX_FU_IN-1];
  logic [DATA_WIDTH-1:0]    obuf_data  [0:NUM_INSTR-1][0:MAX_FU_IN-1];

  // Consume strobe from scheduler fire
  logic                     fire_valid_w;
  logic [SLOT_IDX_W-1:0]   fire_slot_idx_w;
  logic [EFF_OPCODE-1:0]   fire_opcode_w;
  logic [FU_IDX_W-1:0]     fire_fu_idx_w;

  // ---------------------------------------------------------------
  // Operand ingress and buffer management (inline, not submodule)
  //
  // For each PE input that is valid, find matching slot(s) by tag,
  // find operand positions that select this PE input, and if the
  // operand latch is empty, assert ready and capture on transfer.
  // ---------------------------------------------------------------
  generate
    if (!ENABLE_SHARE_OPERAND_BUF) begin : gen_per_instr_buf

      // Ready logic: accept PE input if matching slot/operand is empty
      always_comb begin : buf_ready
        integer iter_var0;
        integer iter_var1;
        integer iter_var2;

        for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : per_pe
          pe_in_ready[iter_var0] = 1'b0;
          if (pe_in_valid[iter_var0]) begin : check
            for (iter_var1 = 0; iter_var1 < NUM_INSTR; iter_var1 = iter_var1 + 1) begin : scan_s
              if (all_slot_valid[iter_var1] &&
                  (all_slot_tag[iter_var1] == pe_in_tag[iter_var0])) begin : tmatch
                for (iter_var2 = 0; iter_var2 < MAX_FU_IN; iter_var2 = iter_var2 + 1) begin : scan_o
                  if (!all_slot_operand_is_reg[iter_var1][iter_var2] &&
                      !all_slot_in_mux_disconnect[iter_var1][iter_var2]) begin : connected
                    if (all_slot_in_mux_discard[iter_var1][iter_var2] &&
                        (EFF_IN_SEL'(iter_var0) == all_slot_in_mux_sel[iter_var1][iter_var2])) begin : accept_discard
                      pe_in_ready[iter_var0] = 1'b1;
                    end : accept_discard
                    else if (!all_slot_in_mux_discard[iter_var1][iter_var2] &&
                             (EFF_IN_SEL'(iter_var0) == all_slot_in_mux_sel[iter_var1][iter_var2]) &&
                             !obuf_valid[iter_var1][iter_var2]) begin : accept_normal
                      pe_in_ready[iter_var0] = 1'b1;
                    end : accept_normal
                  end : connected
                end : scan_o
              end : tmatch
            end : scan_s
          end : check
        end : per_pe
      end : buf_ready

      // Buffer capture and consume (sequential)
      always_ff @(posedge clk or negedge rst_n) begin : buf_seq
        integer iter_var0;
        integer iter_var1;
        integer iter_var2;

        if (!rst_n) begin : buf_reset
          for (iter_var0 = 0; iter_var0 < NUM_INSTR; iter_var0 = iter_var0 + 1) begin : rst_s
            for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : rst_o
              obuf_valid[iter_var0][iter_var1] <= 1'b0;
              obuf_data[iter_var0][iter_var1]  <= '0;
            end : rst_o
          end : rst_s
        end : buf_reset
        else begin : buf_op
          // Consume on fire: clear non-register operands for the fired slot
          if (fire_valid_w) begin : do_consume
            for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : clr_op
              if (iter_var1 < FU_NUM_INPUTS[fire_fu_idx_w]) begin : in_range
                if (!all_slot_operand_is_reg[fire_slot_idx_w][iter_var1] &&
                    !all_slot_in_mux_disconnect[fire_slot_idx_w][iter_var1] &&
                    !all_slot_in_mux_discard[fire_slot_idx_w][iter_var1]) begin : clear_buf
                  obuf_valid[fire_slot_idx_w][iter_var1] <= 1'b0;
                end : clear_buf
              end : in_range
            end : clr_op
          end : do_consume

          // Capture: for each PE input that transfers
          for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : cap_pe
            if (pe_in_valid[iter_var0] && pe_in_ready[iter_var0]) begin : xfer
              for (iter_var1 = 0; iter_var1 < NUM_INSTR; iter_var1 = iter_var1 + 1) begin : cap_s
                if (all_slot_valid[iter_var1] &&
                    (all_slot_tag[iter_var1] == pe_in_tag[iter_var0])) begin : cap_tm
                  for (iter_var2 = 0; iter_var2 < MAX_FU_IN; iter_var2 = iter_var2 + 1) begin : cap_o
                    if (!all_slot_operand_is_reg[iter_var1][iter_var2] &&
                        !all_slot_in_mux_disconnect[iter_var1][iter_var2] &&
                        !all_slot_in_mux_discard[iter_var1][iter_var2] &&
                        (EFF_IN_SEL'(iter_var0) == all_slot_in_mux_sel[iter_var1][iter_var2]) &&
                        !obuf_valid[iter_var1][iter_var2]) begin : do_cap
                      obuf_valid[iter_var1][iter_var2] <= 1'b1;
                      obuf_data[iter_var1][iter_var2]  <= pe_in_data[iter_var0];
                    end : do_cap
                  end : cap_o
                end : cap_tm
              end : cap_s
            end : xfer
          end : cap_pe
        end : buf_op
      end : buf_seq

    end : gen_per_instr_buf
    else begin : gen_shared_buf

      // Shared operand buffer mode: instantiate the operand submodule
      // and expose buffer readiness through its query interface.
      // For scheduling, we query each slot sequentially in comb logic.

      // Shared buffer entries
      logic                     sbuf_used     [0:OPERAND_BUF_SIZE-1];
      logic [TAG_WIDTH-1:0]     sbuf_tag      [0:OPERAND_BUF_SIZE-1];
      logic                     sbuf_op_valid [0:OPERAND_BUF_SIZE-1][0:MAX_FU_IN-1];
      logic [DATA_WIDTH-1:0]    sbuf_op_data  [0:OPERAND_BUF_SIZE-1][0:MAX_FU_IN-1];

      logic [$clog2(OPERAND_BUF_SIZE+1)-1:0] sbuf_used_cnt;

      always_comb begin : sbuf_count
        integer iter_var0;
        sbuf_used_cnt = '0;
        for (iter_var0 = 0; iter_var0 < OPERAND_BUF_SIZE; iter_var0 = iter_var0 + 1) begin : cnt
          if (sbuf_used[iter_var0]) begin : inc
            sbuf_used_cnt = sbuf_used_cnt + 1'b1;
          end : inc
        end : cnt
      end : sbuf_count

      // Ready logic
      always_comb begin : sbuf_ready
        integer iter_var0;
        integer iter_var1;
        integer iter_var2;
        integer iter_var3;
        logic found_slot;
        logic [SLOT_IDX_W-1:0] matched_slot_local;

        for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : per_pe
          pe_in_ready[iter_var0] = 1'b0;
          if (pe_in_valid[iter_var0]) begin : chk
            found_slot = 1'b0;
            matched_slot_local = '0;
            for (iter_var1 = NUM_INSTR-1; iter_var1 >= 0; iter_var1 = iter_var1 - 1) begin : fs
              if (all_slot_valid[iter_var1] &&
                  (all_slot_tag[iter_var1] == pe_in_tag[iter_var0])) begin : fm
                found_slot = 1'b1;
                matched_slot_local = iter_var1[SLOT_IDX_W-1:0];
              end : fm
            end : fs
            if (found_slot) begin : hs
              for (iter_var2 = 0; iter_var2 < MAX_FU_IN; iter_var2 = iter_var2 + 1) begin : co
                if (!all_slot_operand_is_reg[matched_slot_local][iter_var2] &&
                    !all_slot_in_mux_disconnect[matched_slot_local][iter_var2] &&
                    (EFF_IN_SEL'(iter_var0) == all_slot_in_mux_sel[matched_slot_local][iter_var2])) begin : sm
                  if (all_slot_in_mux_discard[matched_slot_local][iter_var2]) begin : disc
                    pe_in_ready[iter_var0] = 1'b1;
                  end : disc
                  else begin : norm
                    // Check if tail entry has free slot or can allocate
                    logic tail_found_local;
                    logic tail_free_local;
                    tail_found_local = 1'b0;
                    tail_free_local = 1'b0;
                    for (iter_var3 = OPERAND_BUF_SIZE-1; iter_var3 >= 0; iter_var3 = iter_var3 - 1) begin : ft
                      if (sbuf_used[iter_var3] &&
                          (sbuf_tag[iter_var3] == pe_in_tag[iter_var0]) &&
                          !tail_found_local) begin : tf
                        tail_found_local = 1'b1;
                        tail_free_local = ~sbuf_op_valid[iter_var3][iter_var2];
                      end : tf
                    end : ft
                    if (!tail_found_local) begin : no_tail
                      pe_in_ready[iter_var0] =
                        (sbuf_used_cnt < OPERAND_BUF_SIZE[$clog2(OPERAND_BUF_SIZE+1)-1:0]);
                    end : no_tail
                    else if (tail_free_local) begin : tf_ok
                      pe_in_ready[iter_var0] = 1'b1;
                    end : tf_ok
                    else begin : tf_full
                      pe_in_ready[iter_var0] =
                        (sbuf_used_cnt < OPERAND_BUF_SIZE[$clog2(OPERAND_BUF_SIZE+1)-1:0]);
                    end : tf_full
                  end : norm
                end : sm
              end : co
            end : hs
          end : chk
        end : per_pe
      end : sbuf_ready

      // Shared buffer sequential logic
      always_ff @(posedge clk or negedge rst_n) begin : sbuf_seq
        integer iter_var0;
        integer iter_var1;
        integer iter_var2;
        integer iter_var3;
        logic found_slot;
        logic [SLOT_IDX_W-1:0] matched_slot_local;
        logic wrote;

        if (!rst_n) begin : sbuf_reset
          for (iter_var0 = 0; iter_var0 < OPERAND_BUF_SIZE; iter_var0 = iter_var0 + 1) begin : rs
            sbuf_used[iter_var0] <= 1'b0;
            sbuf_tag[iter_var0]  <= '0;
            for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : ro
              sbuf_op_valid[iter_var0][iter_var1] <= 1'b0;
              sbuf_op_data[iter_var0][iter_var1]  <= '0;
            end : ro
          end : rs
        end : sbuf_reset
        else begin : sbuf_op
          // Consume on fire: pop head entry for fired slot's tag
          if (fire_valid_w) begin : do_consume
            for (iter_var0 = 0; iter_var0 < OPERAND_BUF_SIZE; iter_var0 = iter_var0 + 1) begin : fh
              if (sbuf_used[iter_var0] &&
                  (sbuf_tag[iter_var0] == all_slot_tag[fire_slot_idx_w])) begin : hm
                sbuf_used[iter_var0] <= 1'b0;
                for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : co
                  sbuf_op_valid[iter_var0][iter_var1] <= 1'b0;
                end : co
                disable fh;
              end : hm
            end : fh
          end : do_consume

          // Capture incoming PE tokens
          for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : cp
            if (pe_in_valid[iter_var0] && pe_in_ready[iter_var0]) begin : xf
              found_slot = 1'b0;
              matched_slot_local = '0;
              for (iter_var1 = NUM_INSTR-1; iter_var1 >= 0; iter_var1 = iter_var1 - 1) begin : fs
                if (all_slot_valid[iter_var1] &&
                    (all_slot_tag[iter_var1] == pe_in_tag[iter_var0])) begin : fm
                  found_slot = 1'b1;
                  matched_slot_local = iter_var1[SLOT_IDX_W-1:0];
                end : fm
              end : fs
              if (found_slot) begin : hs
                for (iter_var2 = 0; iter_var2 < MAX_FU_IN; iter_var2 = iter_var2 + 1) begin : po
                  if (!all_slot_operand_is_reg[matched_slot_local][iter_var2] &&
                      !all_slot_in_mux_disconnect[matched_slot_local][iter_var2] &&
                      !all_slot_in_mux_discard[matched_slot_local][iter_var2] &&
                      (EFF_IN_SEL'(iter_var0) == all_slot_in_mux_sel[matched_slot_local][iter_var2])) begin : wo
                    wrote = 1'b0;
                    // Try tail entry
                    for (iter_var3 = OPERAND_BUF_SIZE-1; iter_var3 >= 0; iter_var3 = iter_var3 - 1) begin : ft
                      if (!wrote && sbuf_used[iter_var3] &&
                          (sbuf_tag[iter_var3] == pe_in_tag[iter_var0]) &&
                          !sbuf_op_valid[iter_var3][iter_var2]) begin : wt
                        sbuf_op_valid[iter_var3][iter_var2] <= 1'b1;
                        sbuf_op_data[iter_var3][iter_var2]  <= pe_in_data[iter_var0];
                        wrote = 1'b1;
                      end : wt
                    end : ft
                    // Allocate new entry
                    if (!wrote) begin : an
                      for (iter_var3 = 0; iter_var3 < OPERAND_BUF_SIZE; iter_var3 = iter_var3 + 1) begin : ff
                        if (!wrote && !sbuf_used[iter_var3]) begin : uf
                          sbuf_used[iter_var3]               <= 1'b1;
                          sbuf_tag[iter_var3]                <= pe_in_tag[iter_var0];
                          sbuf_op_valid[iter_var3][iter_var2] <= 1'b1;
                          sbuf_op_data[iter_var3][iter_var2]  <= pe_in_data[iter_var0];
                          wrote = 1'b1;
                        end : uf
                      end : ff
                    end : an
                  end : wo
                end : po
              end : hs
            end : xf
          end : cp
        end : sbuf_op
      end : sbuf_seq

      // Expose buffer state for readiness checks
      // obuf_valid/obuf_data: peek at FIFO head for each slot's tag
      always_comb begin : sbuf_peek
        integer iter_var0;
        integer iter_var1;
        integer iter_var2;
        logic head_found;

        for (iter_var0 = 0; iter_var0 < NUM_INSTR; iter_var0 = iter_var0 + 1) begin : ps
          for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : po
            obuf_valid[iter_var0][iter_var1] = 1'b0;
            obuf_data[iter_var0][iter_var1]  = '0;
          end : po

          if (all_slot_valid[iter_var0]) begin : sv
            head_found = 1'b0;
            for (iter_var2 = 0; iter_var2 < OPERAND_BUF_SIZE; iter_var2 = iter_var2 + 1) begin : fh
              if (!head_found && sbuf_used[iter_var2] &&
                  (sbuf_tag[iter_var2] == all_slot_tag[iter_var0])) begin : hm
                head_found = 1'b1;
                for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : pk
                  obuf_valid[iter_var0][iter_var1] = sbuf_op_valid[iter_var2][iter_var1];
                  obuf_data[iter_var0][iter_var1]  = sbuf_op_data[iter_var2][iter_var1];
                end : pk
              end : hm
            end : fh
          end : sv
        end : ps
      end : sbuf_peek

    end : gen_shared_buf
  endgenerate

  // ---------------------------------------------------------------
  // Register file
  // ---------------------------------------------------------------
  localparam int unsigned EFF_NUM_REG = (NUM_REG > 0) ? NUM_REG : 1;

  logic [EFF_REG_IDX-1:0] reg_rd_idx  [MAX_FU_IN];
  logic [MAX_FU_IN-1:0]   reg_rd_en;
  logic [DATA_WIDTH-1:0]  reg_rd_data [MAX_FU_IN];
  logic [MAX_FU_IN-1:0]   reg_rd_valid;
  logic [EFF_NUM_REG-1:0] reg_consume_en;

  logic [EFF_REG_IDX-1:0] reg_wr_idx  [MAX_FU_OUT];
  logic [MAX_FU_OUT-1:0]  reg_wr_en;
  logic [DATA_WIDTH-1:0]  reg_wr_data [MAX_FU_OUT];
  logic [MAX_FU_OUT-1:0]  reg_wr_ready;

  fabric_temporal_pe_regfile #(
    .NUM_REG        (NUM_REG),
    .REG_FIFO_DEPTH (REG_FIFO_DEPTH),
    .DATA_WIDTH     (DATA_WIDTH),
    .TAG_WIDTH      (TAG_WIDTH),
    .NUM_RD_PORTS   (MAX_FU_IN),
    .NUM_WR_PORTS   (MAX_FU_OUT)
  ) u_regfile (
    .clk           (clk),
    .rst_n         (rst_n),
    .rd_reg_idx    (reg_rd_idx),
    .rd_enable     (reg_rd_en),
    .rd_data       (reg_rd_data),
    .rd_valid      (reg_rd_valid),
    .rd_consume_en (reg_consume_en),
    .wr_reg_idx    (reg_wr_idx),
    .wr_enable     (reg_wr_en),
    .wr_data       (reg_wr_data),
    .wr_ready      (reg_wr_ready)
  );

  // ---------------------------------------------------------------
  // Scheduler
  // ---------------------------------------------------------------

  // Build per-slot per-operand readiness for the scheduler
  logic sched_operand_buf_rdy [0:NUM_INSTR-1][0:MAX_FU_IN-1];
  logic sched_operand_reg_rdy [0:NUM_INSTR-1][0:MAX_FU_IN-1];

  // For register readiness: query the regfile for each slot's register operands
  // This is done combinationally by checking the regfile's FIFO emptiness
  // for the requested register index.  Since the regfile only has MAX_FU_IN
  // read ports, we multiplex across slots in the scheduler's scan.
  // For a simpler first-cut, we check register non-empty status directly.

  always_comb begin : sched_readiness
    integer iter_var0;
    integer iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_INSTR; iter_var0 = iter_var0 + 1) begin : per_slot
      for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : per_op
        sched_operand_buf_rdy[iter_var0][iter_var1] = obuf_valid[iter_var0][iter_var1];
        sched_operand_reg_rdy[iter_var0][iter_var1] = 1'b0;
      end : per_op
    end : per_slot
  end : sched_readiness

  // Register readiness: for each slot's register operands, check if the
  // register FIFO is non-empty.  We do this by scanning the regfile.
  // NOTE: This requires accessing the regfile's internal FIFO empty flags.
  // We use the regfile's read ports by driving queries for each slot.
  // Since we can only check one slot's registers at a time through the
  // read ports, we use a combinational scan.
  //
  // Simplified approach: For the scheduler, we expose per-register
  // non-empty status from the regfile and check it directly.

  // The regfile's internal FIFO empty flags are not directly exposed.
  // We work around this by using the regfile read ports: for the
  // currently-scanned slot, we check rd_valid (which reflects FIFO
  // non-empty for the queried register).
  //
  // However, the scheduler needs ALL slots checked simultaneously.
  // Real solution: add per-register non-empty output to regfile,
  // then check all_slot_operand_reg_idx against non-empty map.

  // For now, we do a generate-based direct check.  When NUM_REG > 0,
  // we query each register and build a non-empty map.

  generate
    if (NUM_REG > 0) begin : gen_reg_rdy

      // Non-empty map: per register, is the FIFO non-empty?
      logic [NUM_REG-1:0] reg_nonempty;

      // Drive regfile read ports to check each register
      // We need one read port per register to check all simultaneously.
      // Since we only have MAX_FU_IN read ports, we use a different strategy:
      // Query the regfile's internal state directly.
      // Since we can't, we maintain a shadow non-empty tracker.

      // Shadow non-empty tracking: increment on write, decrement on consume
      logic [$clog2(REG_FIFO_DEPTH+1)-1:0] reg_occupancy [0:NUM_REG-1];

      always_ff @(posedge clk or negedge rst_n) begin : reg_occ_seq
        integer iter_var0;
        if (!rst_n) begin : reg_occ_reset
          for (iter_var0 = 0; iter_var0 < NUM_REG; iter_var0 = iter_var0 + 1) begin : ro
            reg_occupancy[iter_var0] <= '0;
          end : ro
        end : reg_occ_reset
        else begin : reg_occ_op
          for (iter_var0 = 0; iter_var0 < NUM_REG; iter_var0 = iter_var0 + 1) begin : uo
            case ({reg_consume_en[iter_var0] && (reg_occupancy[iter_var0] != '0),
                   (reg_wr_en_internal[iter_var0] && (reg_occupancy[iter_var0] < REG_FIFO_DEPTH[$clog2(REG_FIFO_DEPTH+1)-1:0]))})
              2'b10: begin : occ_dec
                reg_occupancy[iter_var0] <= reg_occupancy[iter_var0] - 1'b1;
              end : occ_dec
              2'b01: begin : occ_inc
                reg_occupancy[iter_var0] <= reg_occupancy[iter_var0] + 1'b1;
              end : occ_inc
              default: begin : occ_same
                // No change or simultaneous push/pop
              end : occ_same
            endcase
          end : uo
        end : reg_occ_op
      end : reg_occ_seq

      // Internal write-enable per register (from output arb reg writeback)
      logic [NUM_REG-1:0] reg_wr_en_internal;
      always_comb begin : build_wr_en_internal
        integer iter_var0;
        integer iter_var1;
        reg_wr_en_internal = '0;
        for (iter_var0 = 0; iter_var0 < MAX_FU_OUT; iter_var0 = iter_var0 + 1) begin : per_wp
          if (reg_wr_en[iter_var0]) begin : wp_en
            for (iter_var1 = 0; iter_var1 < NUM_REG; iter_var1 = iter_var1 + 1) begin : scan_r
              if (EFF_REG_IDX'(iter_var1) == reg_wr_idx[iter_var0]) begin : rm
                reg_wr_en_internal[iter_var1] = 1'b1;
              end : rm
            end : scan_r
          end : wp_en
        end : per_wp
      end : build_wr_en_internal

      always_comb begin : build_nonempty
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < NUM_REG; iter_var0 = iter_var0 + 1) begin : ne
          reg_nonempty[iter_var0] = (reg_occupancy[iter_var0] != '0);
        end : ne
      end : build_nonempty

      // Build per-slot per-operand register readiness
      always_comb begin : build_reg_rdy
        integer iter_var0;
        integer iter_var1;
        integer iter_var2;
        for (iter_var0 = 0; iter_var0 < NUM_INSTR; iter_var0 = iter_var0 + 1) begin : per_s
          for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : per_o
            sched_operand_reg_rdy[iter_var0][iter_var1] = 1'b0;
            if (all_slot_operand_is_reg[iter_var0][iter_var1]) begin : is_reg
              for (iter_var2 = 0; iter_var2 < NUM_REG; iter_var2 = iter_var2 + 1) begin : scan_r
                if (EFF_REG_IDX'(iter_var2) == all_slot_operand_reg_idx[iter_var0][iter_var1]) begin : rm
                  sched_operand_reg_rdy[iter_var0][iter_var1] = reg_nonempty[iter_var2];
                end : rm
              end : scan_r
            end : is_reg
          end : per_o
        end : per_s
      end : build_reg_rdy

    end : gen_reg_rdy
  endgenerate

  // FU busy signals (from fu_slot wrappers)
  logic [NUM_FU-1:0] fu_busy;

  // fu_num_inputs as flat array for scheduler
  logic [FU_IN_CNT_W-1:0] fu_num_inputs_flat [NUM_FU];
  always_comb begin : build_fu_num_inputs
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_FU; iter_var0 = iter_var0 + 1) begin : per_fu
      fu_num_inputs_flat[iter_var0] = FU_IN_CNT_W'(FU_NUM_INPUTS[iter_var0]);
    end : per_fu
  end : build_fu_num_inputs

  fabric_temporal_pe_scheduler #(
    .NUM_INSTR  (NUM_INSTR),
    .NUM_FU     (NUM_FU),
    .MAX_FU_IN  (MAX_FU_IN),
    .MAX_FU_OUT (MAX_FU_OUT),
    .NUM_REG    (NUM_REG),
    .NUM_PE_IN  (NUM_IN)
  ) u_scheduler (
    .slot_valid              (all_slot_valid),
    .slot_opcode             (all_slot_opcode),
    .slot_operand_is_reg     (all_slot_operand_is_reg),
    .slot_in_mux_discard     (all_slot_in_mux_discard),
    .slot_in_mux_disconnect  (all_slot_in_mux_disconnect),
    .operand_buf_ready       (sched_operand_buf_rdy),
    .operand_reg_ready       (sched_operand_reg_rdy),
    .fu_busy                 (fu_busy),
    .fu_num_inputs           (fu_num_inputs_flat),
    .fire_valid              (fire_valid_w),
    .fire_slot_idx           (fire_slot_idx_w),
    .fire_opcode             (fire_opcode_w),
    .fire_fu_idx             (fire_fu_idx_w)
  );

  // ---------------------------------------------------------------
  // FU fire and input data drive
  // ---------------------------------------------------------------
  always_comb begin : fu_drive
    integer iter_var0;
    integer iter_var1;

    for (iter_var0 = 0; iter_var0 < NUM_FU; iter_var0 = iter_var0 + 1) begin : per_fu
      fu_fire[iter_var0] = 1'b0;
      for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : per_in
        fu_in_data[iter_var0][iter_var1]  = '0;
        fu_in_valid[iter_var0][iter_var1] = 1'b0;
      end : per_in
    end : per_fu

    if (fire_valid_w) begin : do_fire
      fu_fire[fire_fu_idx_w] = 1'b1;

      // Drive FU input data from operand buffers or registers
      for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : drive_in
        if (all_slot_operand_is_reg[fire_slot_idx_w][iter_var1]) begin : from_reg
          // Data from register read port
          fu_in_data[fire_fu_idx_w][iter_var1]  = reg_rd_data[iter_var1];
          fu_in_valid[fire_fu_idx_w][iter_var1] = reg_rd_valid[iter_var1];
        end : from_reg
        else if (all_slot_in_mux_disconnect[fire_slot_idx_w][iter_var1] ||
                 all_slot_in_mux_discard[fire_slot_idx_w][iter_var1]) begin : skip_in
          // Disconnected or discarded: no data
          fu_in_valid[fire_fu_idx_w][iter_var1] = 1'b0;
        end : skip_in
        else begin : from_buf
          fu_in_data[fire_fu_idx_w][iter_var1]  = obuf_data[fire_slot_idx_w][iter_var1];
          fu_in_valid[fire_fu_idx_w][iter_var1] = obuf_valid[fire_slot_idx_w][iter_var1];
        end : from_buf
      end : drive_in
    end : do_fire
  end : fu_drive

  // Drive regfile read ports for the firing slot's register operands
  always_comb begin : reg_rd_drive
    integer iter_var0;
    reg_rd_en = '0;
    for (iter_var0 = 0; iter_var0 < MAX_FU_IN; iter_var0 = iter_var0 + 1) begin : per_op
      reg_rd_idx[iter_var0] = '0;
      if (fire_valid_w &&
          all_slot_operand_is_reg[fire_slot_idx_w][iter_var0]) begin : rd_reg
        reg_rd_en[iter_var0]  = 1'b1;
        reg_rd_idx[iter_var0] = all_slot_operand_reg_idx[fire_slot_idx_w][iter_var0];
      end : rd_reg
    end : per_op
  end : reg_rd_drive

  // Register consume on fire: pop consumed registers
  always_comb begin : reg_consume_drive
    integer iter_var0;
    integer iter_var1;
    reg_consume_en = '0;
    if (fire_valid_w) begin : do_consume
      for (iter_var0 = 0; iter_var0 < MAX_FU_IN; iter_var0 = iter_var0 + 1) begin : per_op
        if (all_slot_operand_is_reg[fire_slot_idx_w][iter_var0]) begin : is_reg
          // Mark the register for consume
          for (iter_var1 = 0; iter_var1 < EFF_NUM_REG; iter_var1 = iter_var1 + 1) begin : scan_r
            if (iter_var1 < NUM_REG &&
                EFF_REG_IDX'(iter_var1) == all_slot_operand_reg_idx[fire_slot_idx_w][iter_var0]) begin : rm
              reg_consume_en[iter_var1] = 1'b1;
            end : rm
          end : scan_r
        end : is_reg
      end : per_op
    end : do_consume
  end : reg_consume_drive

  // ---------------------------------------------------------------
  // FU slot wrappers (per-FU latency pipeline + interval throttle)
  // ---------------------------------------------------------------
  logic                     pipe_out_valid [0:NUM_FU-1][0:MAX_FU_OUT-1];
  logic [DATA_WIDTH-1:0]    pipe_out_data  [0:NUM_FU-1][0:MAX_FU_OUT-1];
  logic                     out_reg_occ    [0:NUM_FU-1][0:MAX_FU_OUT-1];

  genvar g_fu;
  generate
    for (g_fu = 0; g_fu < NUM_FU; g_fu = g_fu + 1) begin : gen_fu_slot

      // Per-FU output valid/data arrays
      logic [MAX_FU_OUT-1:0]    fu_slot_pipe_valid;
      logic [DATA_WIDTH-1:0]    fu_slot_pipe_data [MAX_FU_OUT];
      logic [MAX_FU_OUT-1:0]    fu_slot_oreg_occ;

      // Pack out_reg_occupied for this FU
      always_comb begin : pack_oreg
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < MAX_FU_OUT; iter_var0 = iter_var0 + 1) begin : po
          fu_slot_oreg_occ[iter_var0] = out_reg_occ[g_fu][iter_var0];
        end : po
      end : pack_oreg

      fabric_temporal_pe_fu_slot #(
        .NUM_FU_IN          (FU_NUM_INPUTS[g_fu]),
        .NUM_FU_OUT         (FU_NUM_OUTPUTS[g_fu]),
        .DATA_WIDTH         (DATA_WIDTH),
        .TAG_WIDTH          (TAG_WIDTH),
        .CONFIGURED_LATENCY (FU_LATENCY[g_fu]),
        .INTRINSIC_LATENCY  (FU_INTRINSIC[g_fu]),
        .CONFIGURED_INTERVAL(FU_INTERVAL[g_fu])
      ) u_fu_slot (
        .clk             (clk),
        .rst_n           (rst_n),
        .fire            (fu_fire[g_fu]),
        .fu_out_valid    (fu_out_valid[g_fu]),
        .fu_out_data     (fu_out_data[g_fu]),
        .pipe_out_valid  (fu_slot_pipe_valid),
        .pipe_out_data   (fu_slot_pipe_data),
        .out_reg_occupied(fu_slot_oreg_occ),
        .busy            (fu_busy[g_fu])
      );

      // Unpack to 2D arrays
      always_comb begin : unpack_pipe
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < MAX_FU_OUT; iter_var0 = iter_var0 + 1) begin : up
          pipe_out_valid[g_fu][iter_var0] = fu_slot_pipe_valid[iter_var0];
          pipe_out_data[g_fu][iter_var0]  = fu_slot_pipe_data[iter_var0];
        end : up
      end : unpack_pipe

    end : gen_fu_slot
  endgenerate

  // ---------------------------------------------------------------
  // Per-FU active slot tracking
  //
  // Each FU remembers which slot it last fired on, for output demux
  // and result config purposes.  An FU becomes inactive when all its
  // output registers are drained.
  // ---------------------------------------------------------------
  logic [NUM_FU-1:0]      fu_active;
  logic [SLOT_IDX_W-1:0]  fu_active_slot [0:NUM_FU-1];

  always_ff @(posedge clk or negedge rst_n) begin : fu_active_seq
    integer iter_var0;
    integer iter_var1;
    logic any_occ;

    if (!rst_n) begin : fu_active_reset
      fu_active <= '0;
      for (iter_var0 = 0; iter_var0 < NUM_FU; iter_var0 = iter_var0 + 1) begin : ra
        fu_active_slot[iter_var0] <= '0;
      end : ra
    end : fu_active_reset
    else begin : fu_active_op
      for (iter_var0 = 0; iter_var0 < NUM_FU; iter_var0 = iter_var0 + 1) begin : per_fu
        // Set active on fire
        if (fire_valid_w && (FU_IDX_W'(iter_var0) == fire_fu_idx_w)) begin : activate
          fu_active[iter_var0]      <= 1'b1;
          fu_active_slot[iter_var0] <= fire_slot_idx_w;
        end : activate
        else begin : check_deactivate
          // Deactivate when all output registers drained and no pipeline inflight
          any_occ = 1'b0;
          for (iter_var1 = 0; iter_var1 < MAX_FU_OUT; iter_var1 = iter_var1 + 1) begin : co
            if (out_reg_occ[iter_var0][iter_var1]) begin : occ
              any_occ = 1'b1;
            end : occ
          end : co
          if (!any_occ && !fu_busy[iter_var0]) begin : deactivate
            fu_active[iter_var0] <= 1'b0;
          end : deactivate
        end : check_deactivate
      end : per_fu
    end : fu_active_op
  end : fu_active_seq

  // ---------------------------------------------------------------
  // Build output_arb config from active slot info
  // ---------------------------------------------------------------
  logic [EFF_OUT_SEL-1:0] arb_out_demux_sel        [0:NUM_FU-1][0:MAX_FU_OUT-1];
  logic                    arb_out_demux_discard    [0:NUM_FU-1][0:MAX_FU_OUT-1];
  logic                    arb_out_demux_disconnect [0:NUM_FU-1][0:MAX_FU_OUT-1];
  logic [TAG_WIDTH-1:0]   arb_result_tag           [0:NUM_FU-1][0:MAX_FU_OUT-1];
  logic                    arb_result_is_reg        [0:NUM_FU-1][0:MAX_FU_OUT-1];
  logic [EFF_REG_IDX-1:0] arb_result_reg_idx       [0:NUM_FU-1][0:MAX_FU_OUT-1];

  always_comb begin : build_arb_cfg
    integer iter_var0;
    integer iter_var1;

    for (iter_var0 = 0; iter_var0 < NUM_FU; iter_var0 = iter_var0 + 1) begin : per_fu
      for (iter_var1 = 0; iter_var1 < MAX_FU_OUT; iter_var1 = iter_var1 + 1) begin : per_out
        if (fu_active[iter_var0]) begin : active
          arb_out_demux_sel[iter_var0][iter_var1]        = all_slot_out_demux_sel[fu_active_slot[iter_var0]][iter_var1];
          arb_out_demux_discard[iter_var0][iter_var1]    = all_slot_out_demux_discard[fu_active_slot[iter_var0]][iter_var1];
          arb_out_demux_disconnect[iter_var0][iter_var1] = all_slot_out_demux_disconnect[fu_active_slot[iter_var0]][iter_var1];
          arb_result_tag[iter_var0][iter_var1]           = all_slot_result_tag[fu_active_slot[iter_var0]][iter_var1];
          arb_result_is_reg[iter_var0][iter_var1]        = all_slot_result_is_reg[fu_active_slot[iter_var0]][iter_var1];
          arb_result_reg_idx[iter_var0][iter_var1]       = all_slot_result_reg_idx[fu_active_slot[iter_var0]][iter_var1];
        end : active
        else begin : inactive
          arb_out_demux_sel[iter_var0][iter_var1]        = '0;
          arb_out_demux_discard[iter_var0][iter_var1]    = 1'b0;
          arb_out_demux_disconnect[iter_var0][iter_var1] = 1'b1;
          arb_result_tag[iter_var0][iter_var1]           = '0;
          arb_result_is_reg[iter_var0][iter_var1]        = 1'b0;
          arb_result_reg_idx[iter_var0][iter_var1]       = '0;
        end : inactive
      end : per_out
    end : per_fu
  end : build_arb_cfg

  // ---------------------------------------------------------------
  // Output arbitration
  // ---------------------------------------------------------------
  logic [MAX_FU_OUT-1:0] arb_reg_wr_en;
  logic [EFF_REG_IDX-1:0] arb_reg_wr_idx  [MAX_FU_OUT];
  logic [DATA_WIDTH-1:0]  arb_reg_wr_data [MAX_FU_OUT];

  fabric_temporal_pe_output_arb #(
    .NUM_FU      (NUM_FU),
    .MAX_FU_OUT  (MAX_FU_OUT),
    .NUM_PE_OUT  (NUM_OUT),
    .DATA_WIDTH  (DATA_WIDTH),
    .TAG_WIDTH   (TAG_WIDTH),
    .NUM_REG     (NUM_REG)
  ) u_output_arb (
    .clk       (clk),
    .rst_n     (rst_n),

    .pipe_valid           (pipe_out_valid),
    .pipe_data            (pipe_out_data),
    .out_reg_occupied     (out_reg_occ),

    .fu_active            (fu_active),
    .fu_out_demux_sel        (arb_out_demux_sel),
    .fu_out_demux_discard    (arb_out_demux_discard),
    .fu_out_demux_disconnect (arb_out_demux_disconnect),
    .fu_result_tag           (arb_result_tag),
    .fu_result_is_reg        (arb_result_is_reg),
    .fu_result_reg_idx       (arb_result_reg_idx),

    .pe_out_valid (pe_out_valid),
    .pe_out_data  (pe_out_data),
    .pe_out_tag   (pe_out_tag),
    .pe_out_ready (pe_out_ready),

    .reg_wr_enable (arb_reg_wr_en),
    .reg_wr_idx    (arb_reg_wr_idx),
    .reg_wr_data   (arb_reg_wr_data)
  );

  // Connect output_arb register writeback to regfile
  always_comb begin : connect_reg_wb
    integer iter_var0;
    reg_wr_en = arb_reg_wr_en;
    for (iter_var0 = 0; iter_var0 < MAX_FU_OUT; iter_var0 = iter_var0 + 1) begin : per_wp
      reg_wr_idx[iter_var0]  = arb_reg_wr_idx[iter_var0];
      reg_wr_data[iter_var0] = arb_reg_wr_data[iter_var0];
    end : per_wp
  end : connect_reg_wb

endmodule : fabric_temporal_pe
