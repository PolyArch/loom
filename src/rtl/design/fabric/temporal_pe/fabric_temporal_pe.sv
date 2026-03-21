// fabric_temporal_pe.sv -- Top-level temporal PE container.
//
// Wires together:
//   - fabric_temporal_pe_imem      (instruction memory + tag-match CAM)
//   - fabric_temporal_pe_regfile   (FIFO-based register file)
//   - fabric_temporal_pe_scheduler (at-most-one FU fires per cycle)
//   - fabric_temporal_pe_fu_slot   (per-FU latency pipeline + interval throttle)
//   - fabric_temporal_pe_output_arb(FU-local output regs + round-robin arbitration)
//
// Operand buffering is inlined in this module for direct visibility by
// the scheduler (which needs per-slot operand readiness for all slots
// simultaneously).
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
  parameter int unsigned TOTAL_FU_CFG_BITS = 0,
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

  // --- PE input ports ---
  input  logic [NUM_IN-1:0]       pe_in_valid,
  output logic [NUM_IN-1:0]       pe_in_ready,
  input  logic [DATA_WIDTH-1:0]   pe_in_data  [NUM_IN],
  input  logic [TAG_WIDTH-1:0]    pe_in_tag   [NUM_IN],

  // --- PE output ports ---
  output logic [NUM_OUT-1:0]      pe_out_valid,
  input  logic [NUM_OUT-1:0]      pe_out_ready,
  output logic [DATA_WIDTH-1:0]   pe_out_data [NUM_OUT],
  output logic [TAG_WIDTH-1:0]    pe_out_tag  [NUM_OUT],

  // --- FU body interface ---
  output logic                     fu_fire     [NUM_FU],
  output logic [DATA_WIDTH-1:0]   fu_in_data  [NUM_FU][MAX_FU_IN],
  output logic [MAX_FU_IN-1:0]    fu_in_valid [NUM_FU],
  input  logic [MAX_FU_OUT-1:0]   fu_out_valid [NUM_FU],
  input  logic [DATA_WIDTH-1:0]   fu_out_data  [NUM_FU][MAX_FU_OUT],

  // --- Persistent FU config bits ---
  output logic [(TOTAL_FU_CFG_BITS > 0 ? TOTAL_FU_CFG_BITS : 1)-1:0] fu_cfg_bits
);

  // ---------------------------------------------------------------
  // Derived widths
  // ---------------------------------------------------------------
  localparam int unsigned SLOT_IDX_W  = (NUM_INSTR > 1) ? $clog2(NUM_INSTR) : 1;
  localparam int unsigned OPCODE_W    = clog2(NUM_FU);
  localparam int unsigned FU_IDX_W    = (NUM_FU > 1) ? $clog2(NUM_FU) : 1;
  localparam int unsigned REG_IDX_W   = clog2(NUM_REG);
  localparam int unsigned IN_SEL_W    = clog2(NUM_IN);
  localparam int unsigned OUT_SEL_W   = clog2(NUM_OUT);
  localparam int unsigned FU_IN_CNT_W = $clog2(MAX_FU_IN + 1);
  localparam int unsigned EFF_OPCODE  = (OPCODE_W  > 0) ? OPCODE_W  : 1;
  localparam int unsigned EFF_REG_IDX = (REG_IDX_W > 0) ? REG_IDX_W : 1;
  localparam int unsigned EFF_IN_SEL  = (IN_SEL_W  > 0) ? IN_SEL_W  : 1;
  localparam int unsigned EFF_OUT_SEL = (OUT_SEL_W > 0) ? OUT_SEL_W : 1;
  localparam int unsigned EFF_NUM_REG = (NUM_REG > 0) ? NUM_REG : 1;

  // ===============================================================
  // IMEM: config store + field decode + tag-match
  // ===============================================================
  logic                    imem_match_found;
  logic [SLOT_IDX_W-1:0]  imem_match_slot;

  // All decoded slot fields from imem
  logic                    all_slot_valid              [0:NUM_INSTR-1];
  logic [TAG_WIDTH-1:0]    all_slot_tag                [0:NUM_INSTR-1];
  logic [EFF_OPCODE-1:0]   all_slot_opcode             [0:NUM_INSTR-1];
  logic [MAX_FU_IN-1:0]   all_slot_operand_is_reg     [0:NUM_INSTR-1];
  logic [EFF_REG_IDX-1:0] all_slot_operand_reg_idx    [0:NUM_INSTR-1][0:MAX_FU_IN-1];
  logic [EFF_IN_SEL-1:0]  all_slot_in_mux_sel         [0:NUM_INSTR-1][0:MAX_FU_IN-1];
  logic [MAX_FU_IN-1:0]   all_slot_in_mux_discard     [0:NUM_INSTR-1];
  logic [MAX_FU_IN-1:0]   all_slot_in_mux_disconnect  [0:NUM_INSTR-1];
  logic [EFF_OUT_SEL-1:0] all_slot_out_demux_sel      [0:NUM_INSTR-1][0:MAX_FU_OUT-1];
  logic [MAX_FU_OUT-1:0]  all_slot_out_demux_discard  [0:NUM_INSTR-1];
  logic [MAX_FU_OUT-1:0]  all_slot_out_demux_disconnect[0:NUM_INSTR-1];
  logic [TAG_WIDTH-1:0]   all_slot_result_tag         [0:NUM_INSTR-1][0:MAX_FU_OUT-1];
  logic [MAX_FU_OUT-1:0]  all_slot_result_is_reg      [0:NUM_INSTR-1];
  logic [EFF_REG_IDX-1:0] all_slot_result_reg_idx     [0:NUM_INSTR-1][0:MAX_FU_OUT-1];

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
    .query_valid (1'b0),
    .query_tag   ('0),
    .match_found    (imem_match_found),
    .match_slot_idx (imem_match_slot),
    .slot_valid              (all_slot_valid),
    .slot_tag                (all_slot_tag),
    .slot_opcode             (all_slot_opcode),
    .slot_operand_is_reg     (all_slot_operand_is_reg),
    .slot_operand_reg_idx    (all_slot_operand_reg_idx),
    .slot_in_mux_sel         (all_slot_in_mux_sel),
    .slot_in_mux_discard     (all_slot_in_mux_discard),
    .slot_in_mux_disconnect  (all_slot_in_mux_disconnect),
    .slot_out_demux_sel      (all_slot_out_demux_sel),
    .slot_out_demux_discard  (all_slot_out_demux_discard),
    .slot_out_demux_disconnect(all_slot_out_demux_disconnect),
    .slot_result_is_reg      (all_slot_result_is_reg),
    .slot_result_reg_idx     (all_slot_result_reg_idx),
    .slot_result_tag         (all_slot_result_tag),
    .fu_cfg_bits             (fu_cfg_bits)
  );

  // ===============================================================
  // SCHEDULER: selects at most one FU to fire per cycle
  // ===============================================================
  logic                    fire_valid_w;
  logic [SLOT_IDX_W-1:0]  fire_slot_idx_w;
  logic [EFF_OPCODE-1:0]  fire_opcode_w;
  logic [FU_IDX_W-1:0]    fire_fu_idx_w;
  logic [NUM_FU-1:0]      fu_busy;

  // Per-slot operand readiness for scheduler
  logic sched_operand_buf_rdy [0:NUM_INSTR-1][0:MAX_FU_IN-1];
  logic sched_operand_reg_rdy [0:NUM_INSTR-1][0:MAX_FU_IN-1];

  logic [FU_IN_CNT_W-1:0] fu_num_inputs_flat [NUM_FU];
  always_comb begin : build_fu_num_inputs
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_FU; iter_var0 = iter_var0 + 1) begin : per_fu
      fu_num_inputs_flat[iter_var0] = FU_IN_CNT_W'(FU_NUM_INPUTS[iter_var0]);
    end : per_fu
  end : build_fu_num_inputs

  // Adapter wires: packed -> unpacked for scheduler 2D array ports
  logic sched_slot_op_is_reg     [0:NUM_INSTR-1][0:MAX_FU_IN-1];
  logic sched_slot_in_discard    [0:NUM_INSTR-1][0:MAX_FU_IN-1];
  logic sched_slot_in_disconnect [0:NUM_INSTR-1][0:MAX_FU_IN-1];

  always_comb begin : build_sched_adapter
    integer iter_var0;
    integer iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_INSTR; iter_var0 = iter_var0 + 1) begin : per_s
      for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : per_o
        sched_slot_op_is_reg[iter_var0][iter_var1]     = all_slot_operand_is_reg[iter_var0][iter_var1];
        sched_slot_in_discard[iter_var0][iter_var1]    = all_slot_in_mux_discard[iter_var0][iter_var1];
        sched_slot_in_disconnect[iter_var0][iter_var1] = all_slot_in_mux_disconnect[iter_var0][iter_var1];
      end : per_o
    end : per_s
  end : build_sched_adapter

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
    .slot_operand_is_reg     (sched_slot_op_is_reg),
    .slot_in_mux_discard     (sched_slot_in_discard),
    .slot_in_mux_disconnect  (sched_slot_in_disconnect),
    .operand_buf_ready       (sched_operand_buf_rdy),
    .operand_reg_ready       (sched_operand_reg_rdy),
    .fu_busy                 (fu_busy),
    .fu_num_inputs           (fu_num_inputs_flat),
    .fire_valid              (fire_valid_w),
    .fire_slot_idx           (fire_slot_idx_w),
    .fire_opcode             (fire_opcode_w),
    .fire_fu_idx             (fire_fu_idx_w)
  );

  // ===============================================================
  // OPERAND BUFFERING (inlined for scheduler visibility)
  // ===============================================================
  logic obuf_valid [0:NUM_INSTR-1][0:MAX_FU_IN-1];
  logic [DATA_WIDTH-1:0] obuf_data [0:NUM_INSTR-1][0:MAX_FU_IN-1];

  generate
    if (!ENABLE_SHARE_OPERAND_BUF) begin : gen_per_instr_buf
      // --- Per-instruction operand latches ---
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
                      !all_slot_in_mux_disconnect[iter_var1][iter_var2]) begin : conn
                    if (all_slot_in_mux_discard[iter_var1][iter_var2] &&
                        (EFF_IN_SEL'(iter_var0) == all_slot_in_mux_sel[iter_var1][iter_var2])) begin : disc_ok
                      pe_in_ready[iter_var0] = 1'b1;
                    end : disc_ok
                    else if (!all_slot_in_mux_discard[iter_var1][iter_var2] &&
                             (EFF_IN_SEL'(iter_var0) == all_slot_in_mux_sel[iter_var1][iter_var2]) &&
                             !obuf_valid[iter_var1][iter_var2]) begin : norm_ok
                      pe_in_ready[iter_var0] = 1'b1;
                    end : norm_ok
                  end : conn
                end : scan_o
              end : tmatch
            end : scan_s
          end : check
        end : per_pe
      end : buf_ready

      always_ff @(posedge clk or negedge rst_n) begin : buf_seq
        integer iter_var0;
        integer iter_var1;
        integer iter_var2;
        if (!rst_n) begin : buf_reset
          for (iter_var0 = 0; iter_var0 < NUM_INSTR; iter_var0 = iter_var0 + 1) begin : rs
            for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : ro
              obuf_valid[iter_var0][iter_var1] <= 1'b0;
              obuf_data[iter_var0][iter_var1]  <= '0;
            end : ro
          end : rs
        end : buf_reset
        else begin : buf_op
          // Consume on fire
          if (fire_valid_w) begin : do_consume
            for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : clr
              if (iter_var1 < FU_NUM_INPUTS[fire_fu_idx_w] &&
                  !all_slot_operand_is_reg[fire_slot_idx_w][iter_var1] &&
                  !all_slot_in_mux_disconnect[fire_slot_idx_w][iter_var1] &&
                  !all_slot_in_mux_discard[fire_slot_idx_w][iter_var1]) begin : clr_buf
                obuf_valid[fire_slot_idx_w][iter_var1] <= 1'b0;
              end : clr_buf
            end : clr
          end : do_consume
          // Capture incoming
          for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : cap
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
          end : cap
        end : buf_op
      end : buf_seq

    end : gen_per_instr_buf

    else begin : gen_shared_buf
      // --- Shared tag-indexed FIFO buffer ---
      logic                    sbuf_used     [0:OPERAND_BUF_SIZE-1];
      logic [TAG_WIDTH-1:0]    sbuf_tag      [0:OPERAND_BUF_SIZE-1];
      logic                    sbuf_op_valid [0:OPERAND_BUF_SIZE-1][0:MAX_FU_IN-1];
      logic [DATA_WIDTH-1:0]   sbuf_op_data  [0:OPERAND_BUF_SIZE-1][0:MAX_FU_IN-1];
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

      always_comb begin : sbuf_ready
        integer iter_var0;
        integer iter_var1;
        integer iter_var2;
        integer iter_var3;
        logic found_slot;
        logic [SLOT_IDX_W-1:0] ms;
        for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : per_pe
          pe_in_ready[iter_var0] = 1'b0;
          if (pe_in_valid[iter_var0]) begin : chk
            found_slot = 1'b0; ms = '0;
            for (iter_var1 = NUM_INSTR-1; iter_var1 >= 0; iter_var1 = iter_var1 - 1) begin : fs
              if (all_slot_valid[iter_var1] &&
                  (all_slot_tag[iter_var1] == pe_in_tag[iter_var0])) begin : fm
                found_slot = 1'b1; ms = iter_var1[SLOT_IDX_W-1:0];
              end : fm
            end : fs
            if (found_slot) begin : hs
              for (iter_var2 = 0; iter_var2 < MAX_FU_IN; iter_var2 = iter_var2 + 1) begin : co
                if (!all_slot_operand_is_reg[ms][iter_var2] &&
                    !all_slot_in_mux_disconnect[ms][iter_var2] &&
                    (EFF_IN_SEL'(iter_var0) == all_slot_in_mux_sel[ms][iter_var2])) begin : sm
                  if (all_slot_in_mux_discard[ms][iter_var2]) begin : dsc
                    pe_in_ready[iter_var0] = 1'b1;
                  end : dsc
                  else begin : nrm
                    logic tf; logic tof;
                    tf = 1'b0; tof = 1'b0;
                    for (iter_var3 = OPERAND_BUF_SIZE-1; iter_var3 >= 0; iter_var3 = iter_var3 - 1) begin : ft
                      if (sbuf_used[iter_var3] &&
                          (sbuf_tag[iter_var3] == pe_in_tag[iter_var0]) && !tf) begin : hit
                        tf = 1'b1;
                        tof = ~sbuf_op_valid[iter_var3][iter_var2];
                      end : hit
                    end : ft
                    if (!tf || tof) begin : can_accept
                      pe_in_ready[iter_var0] = (!tf) ?
                        (sbuf_used_cnt < OPERAND_BUF_SIZE[$clog2(OPERAND_BUF_SIZE+1)-1:0]) : 1'b1;
                    end : can_accept
                    else begin : no_space
                      pe_in_ready[iter_var0] =
                        (sbuf_used_cnt < OPERAND_BUF_SIZE[$clog2(OPERAND_BUF_SIZE+1)-1:0]);
                    end : no_space
                  end : nrm
                end : sm
              end : co
            end : hs
          end : chk
        end : per_pe
      end : sbuf_ready

      always_ff @(posedge clk or negedge rst_n) begin : sbuf_seq
        integer iter_var0;
        integer iter_var1;
        integer iter_var2;
        integer iter_var3;
        logic found_slot;
        logic [SLOT_IDX_W-1:0] ms;
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
          // Consume on fire: pop head for fired slot's tag
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
          // Capture incoming
          for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : cp
            if (pe_in_valid[iter_var0] && pe_in_ready[iter_var0]) begin : xf
              found_slot = 1'b0; ms = '0;
              for (iter_var1 = NUM_INSTR-1; iter_var1 >= 0; iter_var1 = iter_var1 - 1) begin : fs
                if (all_slot_valid[iter_var1] &&
                    (all_slot_tag[iter_var1] == pe_in_tag[iter_var0])) begin : fm
                  found_slot = 1'b1; ms = iter_var1[SLOT_IDX_W-1:0];
                end : fm
              end : fs
              if (found_slot) begin : hs
                for (iter_var2 = 0; iter_var2 < MAX_FU_IN; iter_var2 = iter_var2 + 1) begin : po
                  if (!all_slot_operand_is_reg[ms][iter_var2] &&
                      !all_slot_in_mux_disconnect[ms][iter_var2] &&
                      !all_slot_in_mux_discard[ms][iter_var2] &&
                      (EFF_IN_SEL'(iter_var0) == all_slot_in_mux_sel[ms][iter_var2])) begin : wo
                    wrote = 1'b0;
                    for (iter_var3 = OPERAND_BUF_SIZE-1; iter_var3 >= 0; iter_var3 = iter_var3 - 1) begin : ft
                      if (!wrote && sbuf_used[iter_var3] &&
                          (sbuf_tag[iter_var3] == pe_in_tag[iter_var0]) &&
                          !sbuf_op_valid[iter_var3][iter_var2]) begin : wt
                        sbuf_op_valid[iter_var3][iter_var2] <= 1'b1;
                        sbuf_op_data[iter_var3][iter_var2]  <= pe_in_data[iter_var0];
                        wrote = 1'b1;
                      end : wt
                    end : ft
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

      // Peek at FIFO head for each slot's tag
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

  // ===============================================================
  // SCHEDULER READINESS WIRING
  // ===============================================================
  always_comb begin : sched_buf_readiness
    integer iter_var0;
    integer iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_INSTR; iter_var0 = iter_var0 + 1) begin : per_slot
      for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : per_op
        sched_operand_buf_rdy[iter_var0][iter_var1] = obuf_valid[iter_var0][iter_var1];
      end : per_op
    end : per_slot
  end : sched_buf_readiness

  // ===============================================================
  // REGISTER FILE
  // ===============================================================
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

  // Register operand readiness for scheduler
  generate
    if (NUM_REG > 0) begin : gen_reg_rdy
      // Per-register non-empty tracker (shadow occupancy counter)
      logic [$clog2(REG_FIFO_DEPTH+1)-1:0] reg_occupancy [0:NUM_REG-1];
      logic [NUM_REG-1:0] reg_wr_en_internal;
      logic [NUM_REG-1:0] reg_nonempty;

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

      always_ff @(posedge clk or negedge rst_n) begin : reg_occ_seq
        integer iter_var0;
        if (!rst_n) begin : reg_occ_reset
          for (iter_var0 = 0; iter_var0 < NUM_REG; iter_var0 = iter_var0 + 1) begin : ro
            reg_occupancy[iter_var0] <= '0;
          end : ro
        end : reg_occ_reset
        else begin : reg_occ_op
          for (iter_var0 = 0; iter_var0 < NUM_REG; iter_var0 = iter_var0 + 1) begin : uo
            case ({(reg_consume_en[iter_var0] && (reg_occupancy[iter_var0] != '0)),
                   (reg_wr_en_internal[iter_var0] && (reg_occupancy[iter_var0] < $clog2(REG_FIFO_DEPTH+1)'(REG_FIFO_DEPTH)))})
              2'b10: begin : occ_dec
                reg_occupancy[iter_var0] <= reg_occupancy[iter_var0] - 1'b1;
              end : occ_dec
              2'b01: begin : occ_inc
                reg_occupancy[iter_var0] <= reg_occupancy[iter_var0] + 1'b1;
              end : occ_inc
              default: begin : occ_same
              end : occ_same
            endcase
          end : uo
        end : reg_occ_op
      end : reg_occ_seq

      always_comb begin : build_nonempty
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < NUM_REG; iter_var0 = iter_var0 + 1) begin : ne
          reg_nonempty[iter_var0] = (reg_occupancy[iter_var0] != '0);
        end : ne
      end : build_nonempty

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
    else begin : gen_no_reg_rdy
      always_comb begin : no_reg_rdy
        integer iter_var0;
        integer iter_var1;
        for (iter_var0 = 0; iter_var0 < NUM_INSTR; iter_var0 = iter_var0 + 1) begin : per_s
          for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : per_o
            sched_operand_reg_rdy[iter_var0][iter_var1] = 1'b0;
          end : per_o
        end : per_s
      end : no_reg_rdy
    end : gen_no_reg_rdy
  endgenerate

  // ===============================================================
  // FU FIRE AND INPUT DATA DRIVE
  // ===============================================================
  always_comb begin : fu_drive
    integer iter_var0;
    integer iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_FU; iter_var0 = iter_var0 + 1) begin : per_fu
      fu_fire[iter_var0] = 1'b0;
      fu_in_valid[iter_var0] = '0;
      for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : per_in
        fu_in_data[iter_var0][iter_var1] = '0;
      end : per_in
    end : per_fu
    if (fire_valid_w) begin : do_fire
      fu_fire[fire_fu_idx_w] = 1'b1;
      for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : drive_in
        if (all_slot_operand_is_reg[fire_slot_idx_w][iter_var1]) begin : from_reg
          fu_in_data[fire_fu_idx_w][iter_var1]  = reg_rd_data[iter_var1];
          fu_in_valid[fire_fu_idx_w][iter_var1] = reg_rd_valid[iter_var1];
        end : from_reg
        else if (!all_slot_in_mux_disconnect[fire_slot_idx_w][iter_var1] &&
                 !all_slot_in_mux_discard[fire_slot_idx_w][iter_var1]) begin : from_buf
          fu_in_data[fire_fu_idx_w][iter_var1]  = obuf_data[fire_slot_idx_w][iter_var1];
          fu_in_valid[fire_fu_idx_w][iter_var1] = obuf_valid[fire_slot_idx_w][iter_var1];
        end : from_buf
      end : drive_in
    end : do_fire
  end : fu_drive

  // Drive regfile read ports for fired slot's register operands
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

  // Register consume on fire
  always_comb begin : reg_consume_drive
    integer iter_var0;
    integer iter_var1;
    reg_consume_en = '0;
    if (fire_valid_w) begin : do_consume
      for (iter_var0 = 0; iter_var0 < MAX_FU_IN; iter_var0 = iter_var0 + 1) begin : per_op
        if (all_slot_operand_is_reg[fire_slot_idx_w][iter_var0]) begin : is_reg
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

  // ===============================================================
  // FU SLOT WRAPPERS (per-FU latency pipeline + interval throttle)
  // ===============================================================
  logic pipe_out_valid [0:NUM_FU-1][0:MAX_FU_OUT-1];
  logic [DATA_WIDTH-1:0] pipe_out_data [0:NUM_FU-1][0:MAX_FU_OUT-1];
  logic out_reg_occ [0:NUM_FU-1][0:MAX_FU_OUT-1];

  genvar g_fu;
  generate
    for (g_fu = 0; g_fu < NUM_FU; g_fu = g_fu + 1) begin : gen_fu_slot
      logic [MAX_FU_OUT-1:0] fu_slot_pipe_valid;
      logic [DATA_WIDTH-1:0] fu_slot_pipe_data [MAX_FU_OUT];
      logic [MAX_FU_OUT-1:0] fu_slot_oreg_occ;

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

      always_comb begin : unpack_pipe
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < MAX_FU_OUT; iter_var0 = iter_var0 + 1) begin : up
          pipe_out_valid[g_fu][iter_var0] = fu_slot_pipe_valid[iter_var0];
          pipe_out_data[g_fu][iter_var0]  = fu_slot_pipe_data[iter_var0];
        end : up
      end : unpack_pipe
    end : gen_fu_slot
  endgenerate

  // ===============================================================
  // FU ACTIVE SLOT TRACKING
  // ===============================================================
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
        if (fire_valid_w && (FU_IDX_W'(iter_var0) == fire_fu_idx_w)) begin : activate
          fu_active[iter_var0]      <= 1'b1;
          fu_active_slot[iter_var0] <= fire_slot_idx_w;
        end : activate
        else begin : check_deactivate
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

  // ===============================================================
  // OUTPUT ARB CONFIG FROM ACTIVE SLOT
  // ===============================================================
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

  // ===============================================================
  // OUTPUT ARBITRATION
  // ===============================================================
  logic [MAX_FU_OUT-1:0]  arb_reg_wr_en;
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
