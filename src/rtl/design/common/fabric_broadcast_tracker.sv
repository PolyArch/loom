// fabric_broadcast_tracker.sv -- Broadcast pending-mask tracker.
//
// Tracks per-target acceptance for multi-cycle atomic broadcast.
// A new broadcast token arrives on new_token.  Each target may
// independently accept across multiple cycles via target_accepted.
// The input is consumed (all_accepted goes high) only when ALL
// targets have accepted.
//
// This matches the simulator's broadcastOk_ / per-target accepted
// logic in SimModule.cpp, where an input driving multiple outputs
// is not consumed until every target output has transferred.
//
// When NUM_TARGETS == 1 the module degenerates to a simple
// passthrough (all_accepted = target_accepted[0]).

module fabric_broadcast_tracker #(
  parameter int unsigned NUM_TARGETS = 2
)(
  input  logic                        clk,
  input  logic                        rst_n,

  // Asserted for one cycle when a new broadcast token arrives
  // (i.e. the input has valid data that needs to reach all targets).
  input  logic                        new_token,

  // Per-target acceptance.  Each bit is asserted for one cycle when
  // the corresponding target output completes a handshake transfer
  // for the current broadcast token.
  input  logic [NUM_TARGETS-1:0]      target_accepted,

  // Asserted when all targets have accepted the current token.
  output logic                        all_accepted,

  // Current pending mask -- bit is 1 for targets that have NOT yet
  // accepted.  When all bits are 0, the broadcast is complete.
  output logic [NUM_TARGETS-1:0]      pending
);

  // ---------------------------------------------------------------
  // NUM_TARGETS == 1 edge case
  // ---------------------------------------------------------------
  generate
    if (NUM_TARGETS == 1) begin : gen_single

      // With a single target, acceptance is immediate when the
      // target accepts.  No state needed.
      assign all_accepted = target_accepted[0];
      assign pending      = ~target_accepted;

    end : gen_single

    // ---------------------------------------------------------------
    // General case: NUM_TARGETS >= 2
    // ---------------------------------------------------------------
    else begin : gen_multi

      // accepted_mask[i] is set once target i has accepted for the
      // current broadcast token.  Cleared on new_token or reset.
      logic [NUM_TARGETS-1:0] accepted_mask;

      // A target is still pending if it has not been recorded in
      // accepted_mask AND is not accepting this cycle.
      logic [NUM_TARGETS-1:0] effective_accepted;
      assign effective_accepted = accepted_mask | target_accepted;

      // All targets accepted when every bit is set.
      assign all_accepted = &effective_accepted;

      // Pending is the inverse of effective acceptance.
      assign pending = ~effective_accepted;

      always_ff @(posedge clk or negedge rst_n) begin : mask_update
        if (!rst_n) begin : mask_reset
          accepted_mask <= '0;
        end : mask_reset
        else begin : mask_op
          if (all_accepted || new_token) begin : mask_clear
            // Clear mask for the next token.  If all_accepted and
            // new_token coincide, the new token starts fresh.
            accepted_mask <= '0;
          end : mask_clear
          else begin : mask_accumulate
            // Record any new acceptances this cycle.
            accepted_mask <= effective_accepted;
          end : mask_accumulate
        end : mask_op
      end : mask_update

    end : gen_multi
  endgenerate

endmodule : fabric_broadcast_tracker
