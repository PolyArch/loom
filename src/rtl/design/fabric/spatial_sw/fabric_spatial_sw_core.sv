// fabric_spatial_sw_core.sv -- Non-decomposable spatial switch crossbar.
//
// Per-output mux selecting from connectivity-allowed inputs based on
// the route configuration bitmap.  Supports:
//   - Unicast: one input drives one output
//   - Broadcast: one input drives multiple outputs (multi-cycle atomic
//     consumption via internal broadcast acceptance tracking)
//   - Discard: per-input drain without forwarding
//   - Tagged arbitration: when TAG_WIDTH > 0 and multiple inputs map to
//     the same output, round-robin arbitration selects among valid inputs
//
// Config bitstream layout (low-to-high):
//   [route_bits-1:0]          route bitmap (output-major, input-major
//                              within each output, connected positions only)
//   [route_bits+NUM_IN-1:route_bits]  per-input discard bits
//
// Matches the evaluate/commit behavior in SimModule.cpp SpatialSwitchModule.
//
// Broadcast handling:
//   When one input routes to multiple outputs, the input is consumed
//   only when ALL target outputs have accepted.  Targets may accept
//   across multiple cycles.  A registered accepted_mask tracks which
//   targets have already accepted.  The arbiter request uses only the
//   registered mask (no combinational loop).  The input ready signal
//   uses combinational feedthrough from current-cycle transfers to
//   allow same-cycle consumption when the last target accepts.

module fabric_spatial_sw_core
  import fabric_pkg::*;
#(
  parameter int unsigned NUM_IN    = 2,
  parameter int unsigned NUM_OUT   = 2,
  parameter int unsigned DATA_WIDTH = 32,
  parameter int unsigned TAG_WIDTH  = 0,
  parameter bit [NUM_OUT*NUM_IN-1:0] CONNECTIVITY = {NUM_OUT*NUM_IN{1'b1}}
)(
  input  logic                    clk,
  input  logic                    rst_n,

  // --- Per-input handshake ---
  input  logic [NUM_IN-1:0]       in_valid,
  output logic [NUM_IN-1:0]       in_ready,
  input  logic [DATA_WIDTH-1:0]   in_data  [NUM_IN],
  input  logic [TAG_WIDTH > 0 ? TAG_WIDTH-1 : 0 : 0]
                                  in_tag   [NUM_IN],

  // --- Per-output handshake ---
  output logic [NUM_OUT-1:0]      out_valid,
  input  logic [NUM_OUT-1:0]      out_ready,
  output logic [DATA_WIDTH-1:0]   out_data [NUM_OUT],
  output logic [TAG_WIDTH > 0 ? TAG_WIDTH-1 : 0 : 0]
                                  out_tag  [NUM_OUT],

  // --- Config port (word-serial) ---
  input  logic                    cfg_valid,
  input  logic [31:0]             cfg_wdata,
  output logic                    cfg_ready
);

  // ---------------------------------------------------------------
  // Connectivity helpers
  // ---------------------------------------------------------------

  // Count total connected positions (popcount of CONNECTIVITY).
  function automatic int unsigned count_connected();
    int unsigned cnt;
    integer iter_var0;
    cnt = 0;
    for (iter_var0 = 0; iter_var0 < NUM_OUT * NUM_IN; iter_var0 = iter_var0 + 1) begin : popcount_loop
      if (CONNECTIVITY[iter_var0])
        cnt = cnt + 1;
    end : popcount_loop
    return cnt;
  endfunction : count_connected

  localparam int unsigned ROUTE_BITS  = count_connected();
  localparam int unsigned DISCARD_BITS = NUM_IN;
  localparam int unsigned TOTAL_CFG_BITS = ROUTE_BITS + DISCARD_BITS;
  // Number of 32-bit config words needed.
  localparam int unsigned CFG_WORDS = (TOTAL_CFG_BITS + 31) / 32;

  // Index width for input selection.
  localparam int unsigned IN_IDX_W = $clog2(NUM_IN > 1 ? NUM_IN : 2);

  // ---------------------------------------------------------------
  // Config register storage
  // ---------------------------------------------------------------
  logic [TOTAL_CFG_BITS-1:0] cfg_bits;

  // Config word counter for word-serial loading.
  logic [$clog2(CFG_WORDS > 1 ? CFG_WORDS : 2)-1:0] cfg_word_cnt;

  assign cfg_ready = 1'b1;

  always_ff @(posedge clk or negedge rst_n) begin : cfg_load
    if (!rst_n) begin : cfg_reset
      cfg_bits     <= '0;
      cfg_word_cnt <= '0;
    end : cfg_reset
    else begin : cfg_update
      if (cfg_valid && cfg_ready) begin : cfg_capture
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < 32; iter_var0 = iter_var0 + 1) begin : cfg_bit_loop
          if ((int'(cfg_word_cnt) * 32 + iter_var0) < TOTAL_CFG_BITS) begin : cfg_bit_valid
            cfg_bits[int'(cfg_word_cnt) * 32 + iter_var0] <= cfg_wdata[iter_var0];
          end : cfg_bit_valid
        end : cfg_bit_loop
        // Fabric width adaptation (WA-4): config bit extraction
        // See docs/spec-rtl-width-adaptation.md
        /* verilator lint_off WIDTHTRUNC */
        if (cfg_word_cnt == CFG_WORDS[$clog2(CFG_WORDS > 1 ? CFG_WORDS : 2)-1:0] - 1'b1) begin : cfg_wrap
        /* verilator lint_on WIDTHTRUNC */
          cfg_word_cnt <= '0;
        end : cfg_wrap
        else begin : cfg_incr
          cfg_word_cnt <= cfg_word_cnt + 1'b1;
        end : cfg_incr
      end : cfg_capture
    end : cfg_update
  end : cfg_load

  // Extract route bitmap and discard bits from config storage.
  logic [ROUTE_BITS > 0 ? ROUTE_BITS-1 : 0 : 0] route_bitmap;
  logic [NUM_IN-1:0]                              discard;

  // Fabric width adaptation (WA-4): config bit extraction
  // See docs/spec-rtl-width-adaptation.md
  /* verilator lint_off WIDTHTRUNC */
  generate
    if (ROUTE_BITS > 0) begin : gen_route_extract
      assign route_bitmap = cfg_bits[ROUTE_BITS-1:0];
    end : gen_route_extract
    else begin : gen_route_zero
      assign route_bitmap = '0;
    end : gen_route_zero
  endgenerate
  assign discard = cfg_bits[ROUTE_BITS + NUM_IN - 1 : ROUTE_BITS];
  /* verilator lint_on WIDTHTRUNC */

  // ---------------------------------------------------------------
  // Route-enabled map
  // ---------------------------------------------------------------
  // route_enabled[out][in] = config route bit is set for this pair.
  logic [NUM_IN-1:0] route_enabled [NUM_OUT];

  always_comb begin : build_route_enabled
    integer iter_var0;
    integer iter_var1;
    integer bit_idx;
    bit_idx = 0;
    for (iter_var0 = 0; iter_var0 < NUM_OUT; iter_var0 = iter_var0 + 1) begin : re_out_loop
      route_enabled[iter_var0] = '0;
      for (iter_var1 = 0; iter_var1 < NUM_IN; iter_var1 = iter_var1 + 1) begin : re_in_loop
        if (CONNECTIVITY[iter_var0 * NUM_IN + iter_var1]) begin : re_connected
          if (ROUTE_BITS > 0) begin : re_check_bit
            route_enabled[iter_var0][iter_var1] = route_bitmap[bit_idx];
          end : re_check_bit
          bit_idx = bit_idx + 1;
        end : re_connected
      end : re_in_loop
    end : re_out_loop
  end : build_route_enabled

  // Per-input: target mask (which outputs this input drives).
  logic [NUM_OUT-1:0] input_target_mask [NUM_IN];

  always_comb begin : build_target_mask
    integer iter_var0;
    integer iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : tm_in_loop
      input_target_mask[iter_var0] = '0;
      for (iter_var1 = 0; iter_var1 < NUM_OUT; iter_var1 = iter_var1 + 1) begin : tm_out_loop
        input_target_mask[iter_var0][iter_var1] = route_enabled[iter_var1][iter_var0];
      end : tm_out_loop
    end : tm_in_loop
  end : build_target_mask

  // Per-input: has any target.
  logic [NUM_IN-1:0] input_has_target;

  always_comb begin : compute_has_target
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : ht_loop
      input_has_target[iter_var0] = |input_target_mask[iter_var0];
    end : ht_loop
  end : compute_has_target

  // ---------------------------------------------------------------
  // Broadcast acceptance tracking (registered)
  // ---------------------------------------------------------------
  // Per-input: registered mask of which target outputs have already
  // accepted for the current broadcast token.
  // Bit [out_idx] = 1 means output out_idx has accepted.
  // Non-target positions are always stored as 0 (they are masked
  // out during the all_accepted check).
  logic [NUM_OUT-1:0] accepted_mask_r [NUM_IN];

  // Per-input: which targets still need to accept based on
  // registered state only.  Used for arbiter request generation
  // (no combinational loop).
  // needs_target[in][out] = route_enabled AND NOT accepted_mask_r
  logic [NUM_OUT-1:0] needs_target [NUM_IN];

  always_comb begin : compute_needs_target
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : nt_loop
      needs_target[iter_var0] = input_target_mask[iter_var0] & ~accepted_mask_r[iter_var0];
    end : nt_loop
  end : compute_needs_target

  // ---------------------------------------------------------------
  // Per-output arbitration and mux
  // ---------------------------------------------------------------
  // For non-tagged switches (TAG_WIDTH == 0):
  //   Each output should have at most one route-enabled input in a
  //   valid configuration.  Priority encoder handles degenerate cases.
  //
  // For tagged switches (TAG_WIDTH > 0):
  //   Multiple inputs may be route-enabled for the same output.
  //   Round-robin arbitration selects among valid requesting inputs.

  // Per-output: request vector for arbitration.
  logic [NUM_IN-1:0] out_arb_req [NUM_OUT];

  always_comb begin : build_arb_req
    integer iter_var0;
    integer iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_OUT; iter_var0 = iter_var0 + 1) begin : req_out_loop
      out_arb_req[iter_var0] = '0;
      for (iter_var1 = 0; iter_var1 < NUM_IN; iter_var1 = iter_var1 + 1) begin : req_in_loop
        // Input requests this output if:
        // 1. Route-enabled for this (out, in) pair
        // 2. Input is valid
        // 3. Input is not discarded
        // 4. This target still needs acceptance (registered state only,
        //    no combinational loop through transfer/ready)
        if (route_enabled[iter_var0][iter_var1] &&
            in_valid[iter_var1] &&
            !discard[iter_var1] &&
            needs_target[iter_var1][iter_var0]) begin : req_set
          out_arb_req[iter_var0][iter_var1] = 1'b1;
        end : req_set
      end : req_in_loop
    end : req_out_loop
  end : build_arb_req

  // Per-output arbiter grant (one-hot kept for arbiter interface;
  // only grant_valid and grant_idx are consumed downstream).
  /* verilator lint_off UNUSEDSIGNAL */
  logic [NUM_IN-1:0] out_arb_grant [NUM_OUT];
  /* verilator lint_on UNUSEDSIGNAL */
  logic [NUM_OUT-1:0] out_arb_grant_valid;
  logic [IN_IDX_W-1:0] out_arb_grant_idx [NUM_OUT];

  // Arbiter ack: assert when output transfer completes.
  logic [NUM_OUT-1:0] out_transfer;

  generate
    if (TAG_WIDTH > 0) begin : gen_tagged_arb
      // Tagged: use round-robin arbiter per output.
      genvar go;
      for (go = 0; go < NUM_OUT; go = go + 1) begin : gen_out_arb
        fabric_rr_arbiter #(
          .NUM_REQ(NUM_IN)
        ) u_arb (
          .clk         (clk),
          .rst_n       (rst_n),
          .req         (out_arb_req[go]),
          .ack         (out_transfer[go]),
          .grant       (out_arb_grant[go]),
          .grant_valid (out_arb_grant_valid[go]),
          .grant_idx   (out_arb_grant_idx[go])
        );
      end : gen_out_arb
    end : gen_tagged_arb
    else begin : gen_untagged_arb
      // Non-tagged: priority select (lowest index wins).
      always_comb begin : priority_select
        integer iter_var0;
        integer iter_var1;
        for (iter_var0 = 0; iter_var0 < NUM_OUT; iter_var0 = iter_var0 + 1) begin : ps_out_loop
          out_arb_grant[iter_var0] = '0;
          out_arb_grant_valid[iter_var0] = 1'b0;
          out_arb_grant_idx[iter_var0] = '0;
          for (iter_var1 = 0; iter_var1 < NUM_IN; iter_var1 = iter_var1 + 1) begin : ps_in_loop
            if (!out_arb_grant_valid[iter_var0] && out_arb_req[iter_var0][iter_var1]) begin : ps_hit
              out_arb_grant[iter_var0][iter_var1] = 1'b1;
              out_arb_grant_valid[iter_var0] = 1'b1;
              out_arb_grant_idx[iter_var0] = iter_var1[IN_IDX_W-1:0];
            end : ps_hit
          end : ps_in_loop
        end : ps_out_loop
      end : priority_select
    end : gen_untagged_arb
  endgenerate

  // ---------------------------------------------------------------
  // Output datapath mux
  // ---------------------------------------------------------------
  always_comb begin : output_mux
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_OUT; iter_var0 = iter_var0 + 1) begin : omux_loop
      out_valid[iter_var0] = out_arb_grant_valid[iter_var0];
      out_data[iter_var0]  = in_data[out_arb_grant_idx[iter_var0]];
      out_tag[iter_var0]   = in_tag[out_arb_grant_idx[iter_var0]];
    end : omux_loop
  end : output_mux

  // ---------------------------------------------------------------
  // Output transfer detection
  // ---------------------------------------------------------------
  always_comb begin : detect_transfer
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_OUT; iter_var0 = iter_var0 + 1) begin : xfer_loop
      out_transfer[iter_var0] = out_valid[iter_var0] & out_ready[iter_var0];
    end : xfer_loop
  end : detect_transfer

  // ---------------------------------------------------------------
  // Per-input: current-cycle acceptance (combinational)
  // ---------------------------------------------------------------
  // For each input, compute which targets accept this cycle.
  // A target accepts when the output transfers AND the granted input
  // matches this input.
  logic [NUM_OUT-1:0] cycle_accepted [NUM_IN];

  always_comb begin : compute_cycle_accepted
    integer iter_var0;
    integer iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : ca_in_loop
      cycle_accepted[iter_var0] = '0;
      for (iter_var1 = 0; iter_var1 < NUM_OUT; iter_var1 = iter_var1 + 1) begin : ca_out_loop
        if (out_transfer[iter_var1] &&
            out_arb_grant_idx[iter_var1] == iter_var0[IN_IDX_W-1:0]) begin : ca_match
          cycle_accepted[iter_var0][iter_var1] = 1'b1;
        end : ca_match
      end : ca_out_loop
    end : ca_in_loop
  end : compute_cycle_accepted

  // Effective accepted: combine registered mask with current cycle.
  logic [NUM_OUT-1:0] effective_accepted [NUM_IN];

  always_comb begin : compute_effective_accepted
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : ea_loop
      effective_accepted[iter_var0] = accepted_mask_r[iter_var0] | cycle_accepted[iter_var0];
    end : ea_loop
  end : compute_effective_accepted

  // Per-input: all targets accepted (combinational, for same-cycle
  // consumption when the last target accepts).
  logic [NUM_IN-1:0] all_targets_accepted;

  always_comb begin : compute_all_accepted
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : aa_loop
      // All target positions must be accepted.  Non-target positions
      // (where input_target_mask is 0) are masked out.
      all_targets_accepted[iter_var0] =
        input_has_target[iter_var0] &&
        ((effective_accepted[iter_var0] & input_target_mask[iter_var0]) == input_target_mask[iter_var0]);
    end : aa_loop
  end : compute_all_accepted

  // ---------------------------------------------------------------
  // Accepted mask register update
  // ---------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin : accepted_mask_update
    integer iter_var0;
    if (!rst_n) begin : am_reset
      for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : am_reset_loop
        accepted_mask_r[iter_var0] <= '0;
      end : am_reset_loop
    end : am_reset
    else begin : am_update
      for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : am_update_loop
        if (!in_valid[iter_var0] || all_targets_accepted[iter_var0]) begin : am_clear
          // Token consumed or gone: clear mask for next token.
          accepted_mask_r[iter_var0] <= '0;
        end : am_clear
        else begin : am_accumulate
          // Record any new acceptances this cycle.
          accepted_mask_r[iter_var0] <= effective_accepted[iter_var0];
        end : am_accumulate
      end : am_update_loop
    end : am_update
  end : accepted_mask_update

  // ---------------------------------------------------------------
  // Input ready logic
  // ---------------------------------------------------------------
  // An input is ready when:
  //   - Discarded: always ready (drain)
  //   - No output targets: not ready
  //   - Has targets: ready when all targets have accepted (possibly
  //     combining registered mask and current-cycle transfers)
  always_comb begin : input_ready_logic
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : ir_loop
      if (discard[iter_var0]) begin : ir_discard
        in_ready[iter_var0] = 1'b1;
      end : ir_discard
      else begin : ir_active
        in_ready[iter_var0] = all_targets_accepted[iter_var0];
      end : ir_active
    end : ir_loop
  end : input_ready_logic

endmodule : fabric_spatial_sw_core
