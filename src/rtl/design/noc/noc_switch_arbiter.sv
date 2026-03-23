// noc_switch_arbiter.sv -- Per-output-port round-robin switch arbiter.
//
// Arbitrates among input ports requesting the same output direction.
// For each output port, a round-robin arbiter selects among competing
// input requests.  A grant is only issued if the downstream router
// has credit available for the requested VC on that output port.
//
// Crossbar select outputs indicate which input port feeds each output.

module noc_switch_arbiter
  import noc_pkg::*;
#(
  parameter int unsigned NUM_PORTS = NOC_NUM_PORTS,
  parameter int unsigned NUM_VC    = NOC_NUM_VC_DEFAULT
)(
  input  logic clk,
  input  logic rst_n,

  // Requests from input ports.
  input  logic         req_valid   [NUM_PORTS],
  input  direction_t   req_out_dir [NUM_PORTS],
  input  logic [NOC_VC_ID_WIDTH-1:0] req_vc [NUM_PORTS],

  // Credit availability from downstream (per output port, per VC).
  input  logic [NUM_VC-1:0] downstream_credit [NUM_PORTS],

  // Grants back to input ports.
  output logic         grant       [NUM_PORTS],

  // Crossbar control: per output port, which input port is selected.
  output logic [$clog2(NUM_PORTS)-1:0] xbar_sel   [NUM_PORTS],
  output logic                         xbar_valid [NUM_PORTS]
);

  // ---------------------------------------------------------------
  // Localparams
  // ---------------------------------------------------------------
  localparam int unsigned SEL_W = $clog2(NUM_PORTS);

  // ---------------------------------------------------------------
  // Per-output-port round-robin priority pointer
  // ---------------------------------------------------------------
  logic [SEL_W-1:0] rr_ptr [NUM_PORTS];

  // ---------------------------------------------------------------
  // Arbitration logic (combinational)
  // ---------------------------------------------------------------
  // For each output port, scan input ports in round-robin order.
  // Grant the first eligible request (valid, direction matches,
  // downstream credit available for the requested VC).

  // Intermediate: per-output winner.
  logic [SEL_W-1:0] winner     [NUM_PORTS];
  logic             winner_vld [NUM_PORTS];

  always_comb begin : arb_logic
    integer iter_var0;
    integer iter_var1;

    // Default: no grants, no crossbar selections.
    for (iter_var0 = 0; iter_var0 < NUM_PORTS; iter_var0 = iter_var0 + 1) begin : arb_default_out
      winner[iter_var0]     = '0;
      winner_vld[iter_var0] = 1'b0;
      xbar_sel[iter_var0]   = '0;
      xbar_valid[iter_var0] = 1'b0;
      grant[iter_var0]      = 1'b0;
    end : arb_default_out

    // For each output port (direction), find the winning input.
    for (iter_var0 = 0; iter_var0 < NUM_PORTS; iter_var0 = iter_var0 + 1) begin : arb_per_output
      for (iter_var1 = 0; iter_var1 < NUM_PORTS; iter_var1 = iter_var1 + 1) begin : arb_scan_input
        if (!winner_vld[iter_var0]) begin : arb_check
          automatic logic [SEL_W-1:0] probe_in;
          probe_in = SEL_W'((int'(rr_ptr[iter_var0]) + iter_var1) % NUM_PORTS);

          // Check: input is requesting, targets this output port,
          // and downstream has credit.
          if (req_valid[probe_in]
              && (req_out_dir[probe_in] == direction_t'(3'(iter_var0)))
              && downstream_credit[iter_var0][req_vc[probe_in]]) begin : arb_hit
            winner[iter_var0]     = SEL_W'(probe_in);
            winner_vld[iter_var0] = 1'b1;
          end : arb_hit
        end : arb_check
      end : arb_scan_input
    end : arb_per_output

    // Assign crossbar selects and grants.
    for (iter_var0 = 0; iter_var0 < NUM_PORTS; iter_var0 = iter_var0 + 1) begin : arb_assign
      xbar_sel[iter_var0]   = winner[iter_var0];
      xbar_valid[iter_var0] = winner_vld[iter_var0];
    end : arb_assign

    // Grant: an input port is granted if it won on some output port.
    // An input should only win on at most one output due to the
    // direction uniqueness of XY routing.
    for (iter_var0 = 0; iter_var0 < NUM_PORTS; iter_var0 = iter_var0 + 1) begin : arb_grant_gen
      for (iter_var1 = 0; iter_var1 < NUM_PORTS; iter_var1 = iter_var1 + 1) begin : arb_grant_scan
        if (winner_vld[iter_var1] && (winner[iter_var1] == SEL_W'(iter_var0))) begin : arb_grant_hit
          grant[iter_var0] = 1'b1;
        end : arb_grant_hit
      end : arb_grant_scan
    end : arb_grant_gen
  end : arb_logic

  // ---------------------------------------------------------------
  // Round-robin pointer update (sequential)
  // ---------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin : rr_update
    integer iter_var0;
    if (!rst_n) begin : rr_reset
      for (iter_var0 = 0; iter_var0 < NUM_PORTS; iter_var0 = iter_var0 + 1) begin : rr_reset_loop
        rr_ptr[iter_var0] <= '0;
      end : rr_reset_loop
    end : rr_reset
    else begin : rr_advance
      for (iter_var0 = 0; iter_var0 < NUM_PORTS; iter_var0 = iter_var0 + 1) begin : rr_advance_loop
        if (winner_vld[iter_var0]) begin : rr_advance_hit
          if (winner[iter_var0] == SEL_W'(NUM_PORTS - 1)) begin : rr_wrap
            rr_ptr[iter_var0] <= '0;
          end : rr_wrap
          else begin : rr_incr
            rr_ptr[iter_var0] <= winner[iter_var0] + 1'b1;
          end : rr_incr
        end : rr_advance_hit
      end : rr_advance_loop
    end : rr_advance
  end : rr_update

endmodule : noc_switch_arbiter
