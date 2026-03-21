// fabric_temporal_pe_operand.sv -- Operand routing and buffering.
//
// Two hardware modes selected by ENABLE_SHARE_OPERAND_BUF parameter:
//
// Mode 0 (per-instruction): Each instruction slot has its own single-entry
//   operand latches.  An incoming PE input token is tag-matched to find the
//   slot, then written to the operand position determined by the input mux
//   sel field.  The operand is available until consumed by FU fire.
//
// Mode 1 (shared buffer): One shared operand buffer with OPERAND_BUF_SIZE
//   entries organized as a per-tag FIFO.  Incoming tokens are pushed into
//   the FIFO entry for their tag.  A complete operand row (all required
//   operands present) is dequeued on FU fire.
//
// Each operand may come from a PE input port (via input mux sel) or from
// the register file (via is_reg/reg_idx, handled externally by the top PE).

module fabric_temporal_pe_operand
  import fabric_pkg::*;
#(
  parameter int unsigned NUM_INSTR              = 4,
  parameter int unsigned MAX_FU_IN              = 2,
  parameter int unsigned NUM_PE_IN              = 4,
  parameter int unsigned DATA_WIDTH             = 32,
  parameter int unsigned TAG_WIDTH              = 4,
  parameter bit          ENABLE_SHARE_OPERAND_BUF = 1'b0,
  parameter int unsigned OPERAND_BUF_SIZE       = 8
)(
  input  logic        clk,
  input  logic        rst_n,

  // --- PE input ports ---
  input  logic [NUM_PE_IN-1:0]     pe_in_valid,
  input  logic [DATA_WIDTH-1:0]    pe_in_data    [NUM_PE_IN],
  input  logic [TAG_WIDTH-1:0]     pe_in_tag     [NUM_PE_IN],
  output logic [NUM_PE_IN-1:0]     pe_in_ready,

  // --- Per-operand input mux config (from matched slot) ---
  // These describe, for the currently-queried slot, how each operand
  // maps to a PE input.  Used by ingress capture logic.
  // NOTE: These are provided for ALL slots, not just the matched one.
  //       Index: [slot][operand]
  input  logic [IN_SEL_W-1:0]     slot_in_mux_sel        [0:NUM_INSTR-1][0:MAX_FU_IN-1],
  input  logic                     slot_in_mux_discard    [0:NUM_INSTR-1][0:MAX_FU_IN-1],
  input  logic                     slot_in_mux_disconnect [0:NUM_INSTR-1][0:MAX_FU_IN-1],
  input  logic                     slot_operand_is_reg    [0:NUM_INSTR-1][0:MAX_FU_IN-1],
  input  logic                     slot_valid             [0:NUM_INSTR-1],
  input  logic [TAG_WIDTH-1:0]     slot_tag               [0:NUM_INSTR-1],

  // --- Operand readiness query for a specific slot ---
  input  logic [SLOT_IDX_W-1:0]   query_slot_idx,
  input  logic                     query_valid,
  // Per-operand: is this operand available?
  output logic [MAX_FU_IN-1:0]     operand_ready,
  // Per-operand: peeked data value
  output logic [DATA_WIDTH-1:0]    operand_data [MAX_FU_IN],

  // --- Operand consume strobe (from scheduler on FU fire) ---
  input  logic                     consume_valid,
  input  logic [SLOT_IDX_W-1:0]   consume_slot_idx,
  // Per-operand mask: which operands to consume (register operands excluded)
  input  logic [MAX_FU_IN-1:0]     consume_mask
);

  // ---------------------------------------------------------------
  // Derived widths
  // ---------------------------------------------------------------
  localparam int unsigned SLOT_IDX_W = (NUM_INSTR > 1) ? $clog2(NUM_INSTR) : 1;
  localparam int unsigned IN_SEL_W   = clog2(NUM_PE_IN);
  localparam int unsigned EFF_IN_SEL = (IN_SEL_W > 0) ? IN_SEL_W : 1;

  // ---------------------------------------------------------------
  // Mode 0: Per-instruction operand latches
  // ---------------------------------------------------------------
  generate
    if (!ENABLE_SHARE_OPERAND_BUF) begin : gen_per_instr

      // Per-slot, per-operand: valid flag and data latch
      logic                     buf_valid [0:NUM_INSTR-1][0:MAX_FU_IN-1];
      logic [DATA_WIDTH-1:0]    buf_data  [0:NUM_INSTR-1][0:MAX_FU_IN-1];

      // ---------------------------------------------------------------
      // Ingress: accept PE input tokens and write to operand latches
      // ---------------------------------------------------------------
      // For each PE input, find all slots whose tag matches the input tag.
      // For each such slot, find operand positions that select this PE input.
      // If the operand latch is empty, capture the token.

      // Compute per-PE-input ready: ready if at least one matching slot/operand
      // can accept.
      always_comb begin : ingress_ready
        integer iter_var0;
        integer iter_var1;
        integer iter_var2;

        for (iter_var0 = 0; iter_var0 < NUM_PE_IN; iter_var0 = iter_var0 + 1) begin : per_pe_in
          pe_in_ready[iter_var0] = 1'b0;

          if (pe_in_valid[iter_var0]) begin : check_accept
            for (iter_var1 = 0; iter_var1 < NUM_INSTR; iter_var1 = iter_var1 + 1) begin : scan_slot
              if (slot_valid[iter_var1] && (slot_tag[iter_var1] == pe_in_tag[iter_var0])) begin : tag_match
                for (iter_var2 = 0; iter_var2 < MAX_FU_IN; iter_var2 = iter_var2 + 1) begin : scan_operand
                  if (!slot_operand_is_reg[iter_var1][iter_var2] &&
                      !slot_in_mux_disconnect[iter_var1][iter_var2] &&
                      !slot_in_mux_discard[iter_var1][iter_var2] &&
                      (EFF_IN_SEL'(iter_var0) == slot_in_mux_sel[iter_var1][iter_var2]) &&
                      !buf_valid[iter_var1][iter_var2]) begin : can_accept
                    pe_in_ready[iter_var0] = 1'b1;
                  end : can_accept
                end : scan_operand
                // Also accept if discard mode
                for (iter_var2 = 0; iter_var2 < MAX_FU_IN; iter_var2 = iter_var2 + 1) begin : scan_discard
                  if (!slot_operand_is_reg[iter_var1][iter_var2] &&
                      !slot_in_mux_disconnect[iter_var1][iter_var2] &&
                      slot_in_mux_discard[iter_var1][iter_var2] &&
                      (EFF_IN_SEL'(iter_var0) == slot_in_mux_sel[iter_var1][iter_var2])) begin : discard_accept
                    pe_in_ready[iter_var0] = 1'b1;
                  end : discard_accept
                end : scan_discard
              end : tag_match
            end : scan_slot
          end : check_accept
        end : per_pe_in
      end : ingress_ready

      // Capture transfers into operand buffers
      always_ff @(posedge clk or negedge rst_n) begin : ingress_capture
        integer iter_var0;
        integer iter_var1;
        integer iter_var2;

        if (!rst_n) begin : capture_reset
          for (iter_var0 = 0; iter_var0 < NUM_INSTR; iter_var0 = iter_var0 + 1) begin : rst_slot
            for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : rst_op
              buf_valid[iter_var0][iter_var1] <= 1'b0;
              buf_data[iter_var0][iter_var1]  <= '0;
            end : rst_op
          end : rst_slot
        end : capture_reset
        else begin : capture_op
          // Consume: clear consumed operands
          if (consume_valid) begin : do_consume
            for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : clr_operand
              if (consume_mask[iter_var1]) begin : clr_this
                buf_valid[consume_slot_idx][iter_var1] <= 1'b0;
              end : clr_this
            end : clr_operand
          end : do_consume

          // Capture: for each PE input that transfers, write to matching latches
          for (iter_var0 = 0; iter_var0 < NUM_PE_IN; iter_var0 = iter_var0 + 1) begin : cap_pe_in
            if (pe_in_valid[iter_var0] && pe_in_ready[iter_var0]) begin : transferred
              for (iter_var1 = 0; iter_var1 < NUM_INSTR; iter_var1 = iter_var1 + 1) begin : cap_slot
                if (slot_valid[iter_var1] && (slot_tag[iter_var1] == pe_in_tag[iter_var0])) begin : cap_tag_match
                  for (iter_var2 = 0; iter_var2 < MAX_FU_IN; iter_var2 = iter_var2 + 1) begin : cap_operand
                    if (!slot_operand_is_reg[iter_var1][iter_var2] &&
                        !slot_in_mux_disconnect[iter_var1][iter_var2] &&
                        !slot_in_mux_discard[iter_var1][iter_var2] &&
                        (EFF_IN_SEL'(iter_var0) == slot_in_mux_sel[iter_var1][iter_var2]) &&
                        !buf_valid[iter_var1][iter_var2]) begin : do_capture
                      buf_valid[iter_var1][iter_var2] <= 1'b1;
                      buf_data[iter_var1][iter_var2]  <= pe_in_data[iter_var0];
                    end : do_capture
                  end : cap_operand
                end : cap_tag_match
              end : cap_slot
            end : transferred
          end : cap_pe_in
        end : capture_op
      end : ingress_capture

      // ---------------------------------------------------------------
      // Operand readiness query
      // ---------------------------------------------------------------
      always_comb begin : operand_query
        integer iter_var0;

        for (iter_var0 = 0; iter_var0 < MAX_FU_IN; iter_var0 = iter_var0 + 1) begin : per_operand
          operand_ready[iter_var0] = 1'b0;
          operand_data[iter_var0]  = '0;

          if (query_valid) begin : query_act
            operand_ready[iter_var0] = buf_valid[query_slot_idx][iter_var0];
            operand_data[iter_var0]  = buf_data[query_slot_idx][iter_var0];
          end : query_act
        end : per_operand
      end : operand_query

    end : gen_per_instr

    // ---------------------------------------------------------------
    // Mode 1: Shared tag-indexed FIFO buffer
    // ---------------------------------------------------------------
    else begin : gen_shared

      // Shared buffer: OPERAND_BUF_SIZE entries, each is MAX_FU_IN wide
      // with per-operand valid flags.
      //
      // Implemented as a simple array with tag-indexed access.
      // Each entry has: entry_used, entry_tag, per-operand {valid, data}.
      //
      // Tag matching finds the most recent entry for that tag (FIFO tail)
      // to fill, or allocates a new entry if all operands in the tail are
      // occupied for the needed positions.

      logic                    entry_used    [0:OPERAND_BUF_SIZE-1];
      logic [TAG_WIDTH-1:0]    entry_tag     [0:OPERAND_BUF_SIZE-1];
      logic                    entry_op_valid[0:OPERAND_BUF_SIZE-1][0:MAX_FU_IN-1];
      logic [DATA_WIDTH-1:0]   entry_op_data [0:OPERAND_BUF_SIZE-1][0:MAX_FU_IN-1];

      // Count of used entries
      logic [$clog2(OPERAND_BUF_SIZE+1)-1:0] used_count;

      always_comb begin : count_used
        integer iter_var0;
        used_count = '0;
        for (iter_var0 = 0; iter_var0 < OPERAND_BUF_SIZE; iter_var0 = iter_var0 + 1) begin : cnt_entry
          if (entry_used[iter_var0]) begin : cnt_inc
            used_count = used_count + 1'b1;
          end : cnt_inc
        end : cnt_entry
      end : count_used

      // ---------------------------------------------------------------
      // Find FIFO head and tail for a given tag
      // Head = oldest entry (lowest index among used entries with matching tag)
      // Tail = newest entry (highest index among used entries with matching tag)
      // ---------------------------------------------------------------

      // PE input ready logic
      always_comb begin : shared_ready
        integer iter_var0;
        integer iter_var1;
        integer iter_var2;
        integer iter_var3;
        logic found_slot;
        logic can_write;
        logic [SLOT_IDX_W-1:0] matched_slot;

        for (iter_var0 = 0; iter_var0 < NUM_PE_IN; iter_var0 = iter_var0 + 1) begin : per_pe_in
          pe_in_ready[iter_var0] = 1'b0;

          if (pe_in_valid[iter_var0]) begin : check_pe
            // Find instruction slot for this tag
            found_slot = 1'b0;
            matched_slot = '0;
            for (iter_var1 = NUM_INSTR - 1; iter_var1 >= 0; iter_var1 = iter_var1 - 1) begin : find_slot
              if (slot_valid[iter_var1] && (slot_tag[iter_var1] == pe_in_tag[iter_var0])) begin : found
                found_slot = 1'b1;
                matched_slot = iter_var1[SLOT_IDX_W-1:0];
              end : found
            end : find_slot

            if (found_slot) begin : has_slot
              // Check if any operand position selects this PE input
              can_write = 1'b0;
              for (iter_var2 = 0; iter_var2 < MAX_FU_IN; iter_var2 = iter_var2 + 1) begin : check_operand
                if (!slot_operand_is_reg[matched_slot][iter_var2] &&
                    !slot_in_mux_disconnect[matched_slot][iter_var2] &&
                    (EFF_IN_SEL'(iter_var0) == slot_in_mux_sel[matched_slot][iter_var2])) begin : sel_match
                  if (slot_in_mux_discard[matched_slot][iter_var2]) begin : discard_ok
                    can_write = 1'b1;
                  end : discard_ok
                  else begin : check_buf
                    // Check: find tail entry for this tag, see if operand slot free
                    // or if we can allocate a new entry
                    logic tail_found;
                    logic tail_op_free;
                    tail_found = 1'b0;
                    tail_op_free = 1'b0;
                    for (iter_var3 = OPERAND_BUF_SIZE - 1; iter_var3 >= 0; iter_var3 = iter_var3 - 1) begin : find_tail
                      if (entry_used[iter_var3] && (entry_tag[iter_var3] == pe_in_tag[iter_var0]) && !tail_found) begin : is_tail
                        tail_found = 1'b1;
                        tail_op_free = ~entry_op_valid[iter_var3][iter_var2];
                      end : is_tail
                    end : find_tail
                    if (!tail_found) begin : no_tail
                      // No existing entry: can allocate if space
                      can_write = (used_count < OPERAND_BUF_SIZE[$clog2(OPERAND_BUF_SIZE+1)-1:0]);
                    end : no_tail
                    else if (tail_op_free) begin : tail_free
                      can_write = 1'b1;
                    end : tail_free
                    else begin : tail_full
                      can_write = (used_count < OPERAND_BUF_SIZE[$clog2(OPERAND_BUF_SIZE+1)-1:0]);
                    end : tail_full
                  end : check_buf
                end : sel_match
              end : check_operand
              pe_in_ready[iter_var0] = can_write;
            end : has_slot
          end : check_pe
        end : per_pe_in
      end : shared_ready

      // ---------------------------------------------------------------
      // Capture and consume logic (sequential)
      // ---------------------------------------------------------------
      always_ff @(posedge clk or negedge rst_n) begin : shared_seq
        integer iter_var0;
        integer iter_var1;
        integer iter_var2;
        integer iter_var3;
        logic found_slot;
        logic [SLOT_IDX_W-1:0] matched_slot;

        if (!rst_n) begin : shared_reset
          for (iter_var0 = 0; iter_var0 < OPERAND_BUF_SIZE; iter_var0 = iter_var0 + 1) begin : rst_entry
            entry_used[iter_var0] <= 1'b0;
            entry_tag[iter_var0]  <= '0;
            for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : rst_op
              entry_op_valid[iter_var0][iter_var1] <= 1'b0;
              entry_op_data[iter_var0][iter_var1]  <= '0;
            end : rst_op
          end : rst_entry
        end : shared_reset
        else begin : shared_op
          // Consume: pop head entry for the consumed slot's tag
          if (consume_valid) begin : do_consume
            for (iter_var0 = 0; iter_var0 < OPERAND_BUF_SIZE; iter_var0 = iter_var0 + 1) begin : find_head
              if (entry_used[iter_var0] && (entry_tag[iter_var0] == slot_tag[consume_slot_idx])) begin : head_match
                // This is the head (first used entry for this tag)
                entry_used[iter_var0] <= 1'b0;
                for (iter_var1 = 0; iter_var1 < MAX_FU_IN; iter_var1 = iter_var1 + 1) begin : clr_op
                  entry_op_valid[iter_var0][iter_var1] <= 1'b0;
                end : clr_op
                // Only consume the first matching entry
                disable find_head;
              end : head_match
            end : find_head
          end : do_consume

          // Capture: write incoming PE tokens
          for (iter_var0 = 0; iter_var0 < NUM_PE_IN; iter_var0 = iter_var0 + 1) begin : cap_pe
            if (pe_in_valid[iter_var0] && pe_in_ready[iter_var0]) begin : cap_xfer
              // Find matched instruction slot for this tag
              found_slot = 1'b0;
              matched_slot = '0;
              for (iter_var1 = NUM_INSTR - 1; iter_var1 >= 0; iter_var1 = iter_var1 - 1) begin : find_slot
                if (slot_valid[iter_var1] && (slot_tag[iter_var1] == pe_in_tag[iter_var0])) begin : found
                  found_slot = 1'b1;
                  matched_slot = iter_var1[SLOT_IDX_W-1:0];
                end : found
              end : find_slot

              if (found_slot) begin : has_slot
                for (iter_var2 = 0; iter_var2 < MAX_FU_IN; iter_var2 = iter_var2 + 1) begin : per_op
                  if (!slot_operand_is_reg[matched_slot][iter_var2] &&
                      !slot_in_mux_disconnect[matched_slot][iter_var2] &&
                      !slot_in_mux_discard[matched_slot][iter_var2] &&
                      (EFF_IN_SEL'(iter_var0) == slot_in_mux_sel[matched_slot][iter_var2])) begin : write_op
                    // Find tail entry or allocate new
                    logic wrote;
                    wrote = 1'b0;
                    // Try existing tail entry first
                    for (iter_var3 = OPERAND_BUF_SIZE - 1; iter_var3 >= 0; iter_var3 = iter_var3 - 1) begin : find_tail
                      if (!wrote && entry_used[iter_var3] &&
                          (entry_tag[iter_var3] == pe_in_tag[iter_var0]) &&
                          !entry_op_valid[iter_var3][iter_var2]) begin : write_tail
                        entry_op_valid[iter_var3][iter_var2] <= 1'b1;
                        entry_op_data[iter_var3][iter_var2]  <= pe_in_data[iter_var0];
                        wrote = 1'b1;
                      end : write_tail
                    end : find_tail
                    // If no existing entry, allocate new
                    if (!wrote) begin : alloc_new
                      for (iter_var3 = 0; iter_var3 < OPERAND_BUF_SIZE; iter_var3 = iter_var3 + 1) begin : find_free
                        if (!wrote && !entry_used[iter_var3]) begin : use_free
                          entry_used[iter_var3]               <= 1'b1;
                          entry_tag[iter_var3]                <= pe_in_tag[iter_var0];
                          entry_op_valid[iter_var3][iter_var2] <= 1'b1;
                          entry_op_data[iter_var3][iter_var2]  <= pe_in_data[iter_var0];
                          wrote = 1'b1;
                        end : use_free
                      end : find_free
                    end : alloc_new
                  end : write_op
                end : per_op
              end : has_slot
            end : cap_xfer
          end : cap_pe
        end : shared_op
      end : shared_seq

      // ---------------------------------------------------------------
      // Operand readiness query (peek at FIFO head for tag)
      // ---------------------------------------------------------------
      always_comb begin : shared_query
        integer iter_var0;
        integer iter_var1;
        logic head_found;

        for (iter_var0 = 0; iter_var0 < MAX_FU_IN; iter_var0 = iter_var0 + 1) begin : def_op
          operand_ready[iter_var0] = 1'b0;
          operand_data[iter_var0]  = '0;
        end : def_op

        if (query_valid) begin : query_act
          head_found = 1'b0;
          for (iter_var1 = 0; iter_var1 < OPERAND_BUF_SIZE; iter_var1 = iter_var1 + 1) begin : find_head
            if (!head_found && entry_used[iter_var1] &&
                (entry_tag[iter_var1] == slot_tag[query_slot_idx])) begin : head_match
              head_found = 1'b1;
              for (iter_var0 = 0; iter_var0 < MAX_FU_IN; iter_var0 = iter_var0 + 1) begin : peek_op
                operand_ready[iter_var0] = entry_op_valid[iter_var1][iter_var0];
                operand_data[iter_var0]  = entry_op_data[iter_var1][iter_var0];
              end : peek_op
            end : head_match
          end : find_head
        end : query_act
      end : shared_query

    end : gen_shared
  endgenerate

endmodule : fabric_temporal_pe_operand
