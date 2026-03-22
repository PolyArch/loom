// fabric_temporal_sw_arbiter.sv -- Per-output arbitration with broadcast.
//
// For each output: collect all inputs whose matched slot routes to
// this output.  Round-robin arbitration among competing inputs using
// fabric_rr_arbiter.
//
// Broadcast atomicity: an input targeting multiple outputs is consumed
// only when ALL targets have accepted.  The acceptance tracking is
// functionally equivalent to fabric_broadcast_tracker but implemented
// inline with a two-phase split:
//   - stored accepted mask (accepted_r): registered state from prior
//     cycles, used to gate arbiter requests without combinational loops
//   - effective accepted (accepted_r | this-cycle transfers): used to
//     compute all_accepted for input consumption (in_ready)
// This split avoids the combinational loop that would arise if the
// instantaneous pending mask gated the same arbiter whose grants
// produce the acceptance events.
//
// The connectivity parameter is a packed bit array (output-major,
// input-minor) that defines which (input, output) pairs are
// physically connected.  Route bits from slot_match are packed in
// the same connected-position order and must be expanded to the
// full (NUM_OUT x NUM_IN) space before arbitration.

module fabric_temporal_sw_arbiter
  import fabric_pkg::*;
#(
  parameter int unsigned NUM_IN      = 2,
  parameter int unsigned NUM_OUT     = 2,
  parameter int unsigned DATA_WIDTH  = 32,
  parameter int unsigned TAG_WIDTH   = 4,
  parameter int unsigned ROUTE_BITS  = 4,  // popcount(connectivity)
  // Packed connectivity: bit [o * NUM_IN + i] = 1 if input i can
  // reach output o.  Output-major, input-minor order.
  parameter logic [NUM_OUT*NUM_IN-1:0] CONNECTIVITY = '1
)(
  input  logic                        clk,
  input  logic                        rst_n,

  // --- Per-input signals ---
  input  logic                        in_valid     [0:NUM_IN-1],
  input  logic [DATA_WIDTH-1:0]       in_data      [0:NUM_IN-1],
  input  logic [TAG_WIDTH-1:0]        in_tag       [0:NUM_IN-1],

  // --- Match results from slot_match ---
  input  logic                        match_found  [0:NUM_IN-1],
  input  logic [ROUTE_BITS-1:0]       match_routes [0:NUM_IN-1],

  // --- Output ports ---
  output logic                        out_valid    [0:NUM_OUT-1],
  input  logic                        out_ready    [0:NUM_OUT-1],
  output logic [DATA_WIDTH-1:0]       out_data     [0:NUM_OUT-1],
  output logic [TAG_WIDTH-1:0]        out_tag      [0:NUM_OUT-1],

  // --- Per-input ready (consumed when all broadcast targets accepted) ---
  output logic                        in_ready     [0:NUM_IN-1]
);

  // ---------------------------------------------------------------
  // Expand packed route bits to full (NUM_OUT x NUM_IN) targets
  // ---------------------------------------------------------------
  // For each input, expand its match_routes into a NUM_OUT-wide
  // target vector using the connectivity table.  Route bit ordering
  // matches config packing: output-major, input-minor, only
  // connected positions contribute a bit.

  logic [NUM_OUT-1:0] in_targets [0:NUM_IN-1];

  always_comb begin : expand_routes
    integer iter_var0;
    integer iter_var1;
    integer route_idx;

    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : clear_targets
      in_targets[iter_var0] = '0;
    end : clear_targets

    // For each input, walk the full connectivity table in order and
    // consume route bits.  route_idx is global across the table.
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : per_input_route
      route_idx = 0;
      for (iter_var1 = 0; iter_var1 < NUM_OUT * NUM_IN; iter_var1 = iter_var1 + 1) begin : scan_conn
        if (CONNECTIVITY[iter_var1]) begin : conn_set
          // iter_var1 encodes (outIdx * NUM_IN + inIdx).
          if ((iter_var1 % NUM_IN) == iter_var0) begin : for_this_input
            if (match_found[iter_var0] &&
                match_routes[iter_var0][route_idx]) begin : route_active
              in_targets[iter_var0][iter_var1 / NUM_IN] = 1'b1;
            end : route_active
          end : for_this_input
          route_idx = route_idx + 1;
        end : conn_set
      end : scan_conn
    end : per_input_route
  end : expand_routes

  // ---------------------------------------------------------------
  // Per-input stored accepted mask (registered, no comb loop)
  // ---------------------------------------------------------------
  // accepted_r[i][o] is set when output o has accepted input i's
  // current broadcast token in a previous cycle.  Cleared when the
  // input token changes (valid drops or all targets accepted).
  logic [NUM_OUT-1:0] accepted_r [0:NUM_IN-1];

  // stored_pending[i][o] = target is needed AND not yet accepted.
  // Uses only registered state, safe for combinational request.
  logic [NUM_OUT-1:0] stored_pending [0:NUM_IN-1];

  always_comb begin : calc_stored_pending
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : per_in_pend
      stored_pending[iter_var0] = in_targets[iter_var0] & ~accepted_r[iter_var0];
    end : per_in_pend
  end : calc_stored_pending

  // ---------------------------------------------------------------
  // Per-output: build request vector (no combinational loop)
  // ---------------------------------------------------------------
  logic [NUM_IN-1:0] out_req       [0:NUM_OUT-1];

  always_comb begin : build_req
    integer iter_var0;
    integer iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_OUT; iter_var0 = iter_var0 + 1) begin : per_out_req
      for (iter_var1 = 0; iter_var1 < NUM_IN; iter_var1 = iter_var1 + 1) begin : per_in_req
        out_req[iter_var0][iter_var1] = in_valid[iter_var1]
                                      & match_found[iter_var1]
                                      & stored_pending[iter_var1][iter_var0];
      end : per_in_req
    end : per_out_req
  end : build_req

  // ---------------------------------------------------------------
  // Per-output round-robin arbiters
  // ---------------------------------------------------------------
  logic [NUM_IN-1:0]  out_grant     [0:NUM_OUT-1];
  logic               out_grant_vld [0:NUM_OUT-1];

  localparam int unsigned IN_IDX_W = clog2_min1(NUM_IN);
  logic [IN_IDX_W-1:0] out_winner_idx [0:NUM_OUT-1];

  // Per-output transfer flag.
  logic out_xfer [0:NUM_OUT-1];

  generate
    genvar gi;
    for (gi = 0; gi < NUM_OUT; gi = gi + 1) begin : gen_arb
      logic arb_ack;
      assign arb_ack     = out_valid[gi] & out_ready[gi];
      assign out_xfer[gi] = arb_ack;

      fabric_rr_arbiter #(
        .NUM_REQ (NUM_IN)
      ) u_rr (
        .clk         (clk),
        .rst_n       (rst_n),
        .req         (out_req[gi]),
        .ack         (arb_ack),
        .grant       (out_grant[gi]),
        .grant_valid (out_grant_vld[gi]),
        .grant_idx   (out_winner_idx[gi])
      );
    end : gen_arb
  endgenerate

  // ---------------------------------------------------------------
  // Output data mux
  // ---------------------------------------------------------------
  always_comb begin : out_mux
    integer iter_var0;
    integer iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_OUT; iter_var0 = iter_var0 + 1) begin : per_out_drive
      out_valid[iter_var0] = out_grant_vld[iter_var0];
      out_data[iter_var0]  = '0;
      out_tag[iter_var0]   = '0;
      for (iter_var1 = 0; iter_var1 < NUM_IN; iter_var1 = iter_var1 + 1) begin : per_in_sel
        if (out_grant[iter_var0][iter_var1]) begin : winner_drive
          out_data[iter_var0] = in_data[iter_var1];
          out_tag[iter_var0]  = in_tag[iter_var1];
        end : winner_drive
      end : per_in_sel
    end : per_out_drive
  end : out_mux

  // ---------------------------------------------------------------
  // Broadcast: per-input all_accepted and accepted_r update
  // ---------------------------------------------------------------
  // Compute per-input, per-output "accepted this cycle" flag.
  logic [NUM_OUT-1:0] in_accepted_now [0:NUM_IN-1];

  always_comb begin : calc_accepted_now
    integer iter_var0;
    integer iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : per_in_acc
      for (iter_var1 = 0; iter_var1 < NUM_OUT; iter_var1 = iter_var1 + 1) begin : per_out_acc
        in_accepted_now[iter_var0][iter_var1] = out_xfer[iter_var1]
                                              & out_grant[iter_var1][iter_var0];
      end : per_out_acc
    end : per_in_acc
  end : calc_accepted_now

  // Effective accepted = stored OR this-cycle.
  logic [NUM_OUT-1:0] effective_accepted [0:NUM_IN-1];

  always_comb begin : calc_effective
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : per_in_eff
      effective_accepted[iter_var0] = accepted_r[iter_var0] | in_accepted_now[iter_var0];
    end : per_in_eff
  end : calc_effective

  // all_accepted: input can be consumed when all targets accepted.
  logic all_accepted [0:NUM_IN-1];

  always_comb begin : calc_broadcast_ok
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : per_in_ok
      if (in_valid[iter_var0] && match_found[iter_var0] && (|in_targets[iter_var0])) begin : has_targets
        // All target bits that are set in in_targets must be accepted.
        all_accepted[iter_var0] = &(effective_accepted[iter_var0] | ~in_targets[iter_var0]);
      end : has_targets
      else begin : no_targets
        all_accepted[iter_var0] = 1'b0;
      end : no_targets
    end : per_in_ok
  end : calc_broadcast_ok

  // ---------------------------------------------------------------
  // Input ready
  // ---------------------------------------------------------------
  always_comb begin : in_ready_logic
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : per_in_rdy
      in_ready[iter_var0] = all_accepted[iter_var0];
    end : per_in_rdy
  end : in_ready_logic

  // ---------------------------------------------------------------
  // accepted_r register update
  // ---------------------------------------------------------------
  // On clock edge: if the input is consumed (all_accepted) or input
  // becomes invalid, clear the mask.  Otherwise accumulate.
  always_ff @(posedge clk or negedge rst_n) begin : accepted_update
    integer iter_var0;
    if (!rst_n) begin : acc_reset
      for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : acc_reset_loop
        accepted_r[iter_var0] <= '0;
      end : acc_reset_loop
    end : acc_reset
    else begin : acc_op
      for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : acc_per_input
        if (!in_valid[iter_var0] || !match_found[iter_var0] || all_accepted[iter_var0]) begin : acc_clear
          accepted_r[iter_var0] <= '0;
        end : acc_clear
        else begin : acc_accumulate
          accepted_r[iter_var0] <= effective_accepted[iter_var0];
        end : acc_accumulate
      end : acc_per_input
    end : acc_op
  end : accepted_update

endmodule : fabric_temporal_sw_arbiter
