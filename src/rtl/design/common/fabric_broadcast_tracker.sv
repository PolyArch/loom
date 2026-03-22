// fabric_broadcast_tracker.sv -- Broadcast pending-mask tracker.
//
// Tracks per-target acceptance for multi-cycle atomic broadcast.
// The tracker is active only when token_valid is asserted, indicating
// the input port has a live token that needs to reach all targets.
// Each target may independently accept across multiple cycles via
// target_accepted.  The input may be consumed (all_accepted goes high)
// only when ALL targets have accepted the current token.
//
// When a token is consumed (all_accepted) or token_valid drops, the
// accumulated acceptance mask resets for the next token.
//
// This matches the simulator's broadcastOk_ / per-target accepted
// logic in SimModule.cpp, where readiness is computed only for a
// currently valid input token and acceptance is tracked per-token.

module fabric_broadcast_tracker #(
  parameter int unsigned NUM_TARGETS = 2
)(
  input  logic                        clk,
  input  logic                        rst_n,

  // Asserted when the input port has a valid broadcast token.
  // When deasserted, no acceptance tracking occurs.
  input  logic                        token_valid,

  // Per-target acceptance.  Each bit is asserted for one cycle when
  // the corresponding target output completes a handshake transfer
  // for the current broadcast token.
  input  logic [NUM_TARGETS-1:0]      target_accepted,

  // Asserted when all targets have accepted the current token.
  // Only meaningful when token_valid is high.
  output logic                        all_accepted,

  // Current pending mask -- bit is 1 for targets that have NOT yet
  // accepted.  Only meaningful when token_valid is high.
  output logic [NUM_TARGETS-1:0]      pending
);

  // ---------------------------------------------------------------
  // NUM_TARGETS == 1 edge case
  // ---------------------------------------------------------------
  generate
    if (NUM_TARGETS == 1) begin : gen_single

      // Single target: acceptance is immediate, no state needed.
      // all_accepted is gated by token_valid for correct semantics.
      assign all_accepted = token_valid & target_accepted[0];
      assign pending      = token_valid ? ~target_accepted : '0;

    end : gen_single

    // ---------------------------------------------------------------
    // General case: NUM_TARGETS >= 2
    // ---------------------------------------------------------------
    else begin : gen_multi

      // accepted_mask[i] is set once target i has accepted for the
      // current broadcast token.  Cleared when the token is consumed
      // (all_accepted) or when token_valid drops (token gone).
      logic [NUM_TARGETS-1:0] accepted_mask;

      // Combine stored and instantaneous acceptance.
      logic [NUM_TARGETS-1:0] effective_accepted;
      assign effective_accepted = accepted_mask | target_accepted;

      // All targets accepted when every bit is set AND token is valid.
      assign all_accepted = token_valid & (&effective_accepted);

      // Pending: which targets still need to accept (only when active).
      assign pending = token_valid ? ~effective_accepted : '0;

      always_ff @(posedge clk or negedge rst_n) begin : mask_update
        if (!rst_n) begin : mask_reset
          accepted_mask <= '0;
        end : mask_reset
        else begin : mask_op
          if (!token_valid || all_accepted) begin : mask_clear
            // No active token or token consumed: reset for next token.
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
