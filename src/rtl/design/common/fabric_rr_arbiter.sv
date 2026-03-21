// fabric_rr_arbiter.sv -- Round-robin arbiter.
//
// Selects one requestor per cycle in round-robin order.  The
// round-robin pointer advances when the granted request is
// acknowledged (ack asserted).
//
// When NUM_REQ == 1 the module degenerates to a simple passthrough.

module fabric_rr_arbiter #(
  parameter int unsigned NUM_REQ = 2
)(
  input  logic                              clk,
  input  logic                              rst_n,

  // Request vector -- one bit per requestor.
  input  logic [NUM_REQ-1:0]                req,

  // Acknowledge -- assert for one cycle when the granted request is
  // consumed downstream.  The rr_pointer advances on ack.
  input  logic                              ack,

  // One-hot grant vector.
  output logic [NUM_REQ-1:0]                grant,

  // Asserted when any request is granted.
  output logic                              grant_valid,

  // Binary index of the granted requestor.
  output logic [$clog2(NUM_REQ > 1 ? NUM_REQ : 2)-1:0] grant_idx
);

  // ---------------------------------------------------------------
  // Localparams
  // ---------------------------------------------------------------
  localparam int unsigned IDX_W = $clog2(NUM_REQ > 1 ? NUM_REQ : 2);

  // ---------------------------------------------------------------
  // NUM_REQ == 1 edge case -- trivial passthrough
  // ---------------------------------------------------------------
  generate
    if (NUM_REQ == 1) begin : gen_single

      assign grant       = req;
      assign grant_valid = req[0];
      assign grant_idx   = '0;

    end : gen_single
    else begin : gen_rr

      // -------------------------------------------------------------
      // Round-robin pointer -- points to the next requestor to be
      // given highest priority.
      // -------------------------------------------------------------
      logic [IDX_W-1:0] rr_pointer;

      always_ff @(posedge clk or negedge rst_n) begin : rr_pointer_update
        if (!rst_n) begin : rr_pointer_reset
          rr_pointer <= '0;
        end : rr_pointer_reset
        else if (ack && grant_valid) begin : rr_pointer_advance
          // Advance to one past the current winner.
          if (grant_idx == NUM_REQ[IDX_W-1:0] - 1'b1) begin : rr_pointer_wrap
            rr_pointer <= '0;
          end : rr_pointer_wrap
          else begin : rr_pointer_incr
            rr_pointer <= grant_idx + 1'b1;
          end : rr_pointer_incr
        end : rr_pointer_advance
      end : rr_pointer_update

      // -------------------------------------------------------------
      // Combinational priority scan starting from rr_pointer.
      //
      // We build a doubled request vector and scan from rr_pointer
      // through rr_pointer + NUM_REQ - 1.  The first set bit wins.
      // -------------------------------------------------------------
      logic [2*NUM_REQ-1:0] doubled_req;
      assign doubled_req = {req, req};

      // Intermediate scan results.
      logic [NUM_REQ-1:0]  grant_comb;
      logic                grant_valid_comb;
      logic [IDX_W-1:0]    grant_idx_comb;

      always_comb begin : rr_scan
        integer iter_var0;
        grant_comb      = '0;
        grant_valid_comb = 1'b0;
        grant_idx_comb  = '0;

        for (iter_var0 = 0; iter_var0 < NUM_REQ; iter_var0 = iter_var0 + 1) begin : rr_scan_loop
          if (!grant_valid_comb) begin : rr_scan_check
            // Compute the actual requestor index with wrap-around.
            automatic int unsigned probe_idx;
            probe_idx = (int'(rr_pointer) + iter_var0) % NUM_REQ;
            if (doubled_req[int'(rr_pointer) + iter_var0]) begin : rr_scan_hit
              grant_comb[probe_idx] = 1'b1;
              grant_valid_comb       = 1'b1;
              grant_idx_comb        = probe_idx[IDX_W-1:0];
            end : rr_scan_hit
          end : rr_scan_check
        end : rr_scan_loop
      end : rr_scan

      assign grant       = grant_comb;
      assign grant_valid = grant_valid_comb;
      assign grant_idx   = grant_idx_comb;

    end : gen_rr
  endgenerate

endmodule : fabric_rr_arbiter
