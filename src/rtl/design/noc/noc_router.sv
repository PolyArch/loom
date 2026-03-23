// noc_router.sv -- Parameterized NoC router with 2-stage pipeline.
//
// Implements a wormhole-switched router with XY dimension-ordered
// routing, virtual channels, and credit-based flow control.
//
// Pipeline stages:
//   RC (Route Compute) -- performed inside noc_input_port on HEAD/SINGLE
//   ST (Switch Traversal) -- arbiter + crossbar, registered output
//
// Ports: 5 directional ports (NORTH, EAST, SOUTH, WEST, LOCAL).
// Each port has per-VC input buffers, credit counters for downstream
// flow control, and valid/ready handshaking.

module noc_router
  import noc_pkg::*;
#(
  parameter int unsigned DATA_WIDTH   = NOC_DATA_WIDTH_DEFAULT,
  parameter int unsigned NUM_VC       = NOC_NUM_VC_DEFAULT,
  parameter int unsigned BUFFER_DEPTH = NOC_BUFFER_DEPTH_DEFAULT,
  parameter int unsigned NUM_PORTS    = NOC_NUM_PORTS,
  parameter int unsigned ROUTER_ROW   = 0,
  parameter int unsigned ROUTER_COL   = 0,
  /* verilator lint_off UNUSEDPARAM */
  parameter int unsigned MESH_ROWS    = 2,
  /* verilator lint_on UNUSEDPARAM */
  parameter int unsigned MESH_COLS    = 2
)(
  input  logic clk,
  input  logic rst_n,

  // Input ports (from neighbors and local core).
  input  logic [flit_width(DATA_WIDTH)-1:0] in_flit  [NUM_PORTS],
  input  logic                              in_valid [NUM_PORTS],
  output logic                              in_ready [NUM_PORTS],

  // Output ports (to neighbors and local core).
  // Note: out_ready is retained for interface compatibility but is not
  // used internally; flow control is credit-based via credit_in.
  output logic [flit_width(DATA_WIDTH)-1:0] out_flit  [NUM_PORTS],
  output logic                              out_valid [NUM_PORTS],
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic                              out_ready [NUM_PORTS],
  /* verilator lint_on UNUSEDSIGNAL */

  // Credit interface.
  output logic [NUM_VC-1:0] credit_out [NUM_PORTS],  // credits to upstream
  input  logic [NUM_VC-1:0] credit_in  [NUM_PORTS]   // credits from downstream
);

  // ---------------------------------------------------------------
  // Localparams
  // ---------------------------------------------------------------
  localparam int unsigned FLIT_W = flit_width(DATA_WIDTH);
  localparam int unsigned SEL_W  = $clog2(NUM_PORTS);

  // ---------------------------------------------------------------
  // Input port instances
  // ---------------------------------------------------------------
  // Per-input-port route-compute outputs.
  logic                              ip_rc_valid   [NUM_PORTS];
  direction_t                        ip_rc_out_dir [NUM_PORTS];
  logic [NOC_VC_ID_WIDTH-1:0]        ip_rc_vc      [NUM_PORTS];
  logic [FLIT_W-1:0]                 ip_rc_flit    [NUM_PORTS];
  logic                              ip_grant      [NUM_PORTS];
  logic [NUM_VC-1:0]                 ip_credit_out [NUM_PORTS];

  generate
    genvar gv_ip;
    for (gv_ip = 0; gv_ip < NUM_PORTS; gv_ip = gv_ip + 1) begin : gen_input_port
      noc_input_port #(
        .DATA_WIDTH   (DATA_WIDTH),
        .NUM_VC       (NUM_VC),
        .BUFFER_DEPTH (BUFFER_DEPTH),
        .ROUTER_ROW   (ROUTER_ROW),
        .ROUTER_COL   (ROUTER_COL),
        .MESH_COLS    (MESH_COLS)
      ) u_input_port (
        .clk        (clk),
        .rst_n      (rst_n),
        .flit_in    (in_flit[gv_ip]),
        .flit_valid (in_valid[gv_ip]),
        .flit_ready (in_ready[gv_ip]),
        .rc_valid   (ip_rc_valid[gv_ip]),
        .rc_out_dir (ip_rc_out_dir[gv_ip]),
        .rc_vc      (ip_rc_vc[gv_ip]),
        .rc_flit    (ip_rc_flit[gv_ip]),
        .grant      (ip_grant[gv_ip]),
        .credit_out (ip_credit_out[gv_ip])
      );
    end : gen_input_port
  endgenerate

  // Forward credit_out from input ports to upstream.
  always_comb begin : credit_out_fwd
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_PORTS; iter_var0 = iter_var0 + 1) begin : credit_fwd_loop
      credit_out[iter_var0] = ip_credit_out[iter_var0];
    end : credit_fwd_loop
  end : credit_out_fwd

  // ---------------------------------------------------------------
  // Downstream credit counters
  // ---------------------------------------------------------------
  // Track available buffer space at each downstream neighbor.
  // Initialized to BUFFER_DEPTH; decremented on send, incremented
  // on credit_in pulse.
  localparam int unsigned CREDIT_CNT_W = $clog2(BUFFER_DEPTH + 1);

  logic [CREDIT_CNT_W-1:0] ds_credit_cnt [NUM_PORTS][NUM_VC];
  logic [NUM_VC-1:0]        ds_credit_avail [NUM_PORTS];

  always_ff @(posedge clk or negedge rst_n) begin : ds_credit_mgmt
    integer iter_var0;
    integer iter_var1;
    if (!rst_n) begin : ds_credit_reset
      for (iter_var0 = 0; iter_var0 < NUM_PORTS; iter_var0 = iter_var0 + 1) begin : ds_reset_port
        for (iter_var1 = 0; iter_var1 < NUM_VC; iter_var1 = iter_var1 + 1) begin : ds_reset_vc
          ds_credit_cnt[iter_var0][iter_var1] <= CREDIT_CNT_W'(BUFFER_DEPTH);
        end : ds_reset_vc
      end : ds_reset_port
    end : ds_credit_reset
    else begin : ds_credit_update
      for (iter_var0 = 0; iter_var0 < NUM_PORTS; iter_var0 = iter_var0 + 1) begin : ds_update_port
        for (iter_var1 = 0; iter_var1 < NUM_VC; iter_var1 = iter_var1 + 1) begin : ds_update_vc
          // Determine if we are sending a flit to this output port on this VC.
          automatic logic sending;
          automatic logic receiving_credit;
          sending = st_out_valid[iter_var0]
                    && (st_out_vc[iter_var0] == NOC_VC_ID_WIDTH'(iter_var1));
          receiving_credit = credit_in[iter_var0][iter_var1];

          if (sending && !receiving_credit) begin : ds_decrement
            ds_credit_cnt[iter_var0][iter_var1]
              <= ds_credit_cnt[iter_var0][iter_var1] - 1'b1;
          end : ds_decrement
          else if (!sending && receiving_credit) begin : ds_increment
            ds_credit_cnt[iter_var0][iter_var1]
              <= ds_credit_cnt[iter_var0][iter_var1] + 1'b1;
          end : ds_increment
          // If both sending and receiving, count stays the same.
        end : ds_update_vc
      end : ds_update_port
    end : ds_credit_update
  end : ds_credit_mgmt

  // Credit availability: at least one slot available.
  always_comb begin : ds_credit_avail_gen
    integer iter_var0;
    integer iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_PORTS; iter_var0 = iter_var0 + 1) begin : ds_avail_port
      for (iter_var1 = 0; iter_var1 < NUM_VC; iter_var1 = iter_var1 + 1) begin : ds_avail_vc
        ds_credit_avail[iter_var0][iter_var1]
          = (ds_credit_cnt[iter_var0][iter_var1] != '0);
      end : ds_avail_vc
    end : ds_avail_port
  end : ds_credit_avail_gen

  // ---------------------------------------------------------------
  // Switch arbiter
  // ---------------------------------------------------------------
  logic [SEL_W-1:0] arb_xbar_sel   [NUM_PORTS];
  logic              arb_xbar_valid [NUM_PORTS];

  noc_switch_arbiter #(
    .NUM_PORTS (NUM_PORTS),
    .NUM_VC    (NUM_VC)
  ) u_arbiter (
    .clk              (clk),
    .rst_n            (rst_n),
    .req_valid        (ip_rc_valid),
    .req_out_dir      (ip_rc_out_dir),
    .req_vc           (ip_rc_vc),
    .downstream_credit(ds_credit_avail),
    .grant            (ip_grant),
    .xbar_sel         (arb_xbar_sel),
    .xbar_valid       (arb_xbar_valid)
  );

  // ---------------------------------------------------------------
  // Crossbar
  // ---------------------------------------------------------------
  logic [FLIT_W-1:0] xbar_out_flits [NUM_PORTS];
  logic               xbar_out_valid [NUM_PORTS];

  noc_crossbar #(
    .DATA_WIDTH (DATA_WIDTH),
    .NUM_PORTS  (NUM_PORTS)
  ) u_crossbar (
    .in_flits  (ip_rc_flit),
    .sel       (arb_xbar_sel),
    .sel_valid (arb_xbar_valid),
    .out_flits (xbar_out_flits),
    .out_valid (xbar_out_valid)
  );

  // ---------------------------------------------------------------
  // ST stage: register crossbar outputs
  // ---------------------------------------------------------------
  logic [FLIT_W-1:0] st_out_flit  [NUM_PORTS];
  logic               st_out_valid [NUM_PORTS];
  logic [NOC_VC_ID_WIDTH-1:0] st_out_vc [NUM_PORTS];

  // Capture the VC of the winning input for each output port
  // (needed for credit decrement).
  logic [NOC_VC_ID_WIDTH-1:0] xbar_out_vc [NUM_PORTS];

  always_comb begin : xbar_vc_extract
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_PORTS; iter_var0 = iter_var0 + 1) begin : vc_extract_loop
      if (arb_xbar_valid[iter_var0]) begin : vc_from_winner
        xbar_out_vc[iter_var0] = ip_rc_vc[arb_xbar_sel[iter_var0]];
      end : vc_from_winner
      else begin : vc_default
        xbar_out_vc[iter_var0] = '0;
      end : vc_default
    end : vc_extract_loop
  end : xbar_vc_extract

  always_ff @(posedge clk or negedge rst_n) begin : st_register
    integer iter_var0;
    if (!rst_n) begin : st_reset
      for (iter_var0 = 0; iter_var0 < NUM_PORTS; iter_var0 = iter_var0 + 1) begin : st_reset_loop
        st_out_flit[iter_var0]  <= '0;
        st_out_valid[iter_var0] <= 1'b0;
        st_out_vc[iter_var0]    <= '0;
      end : st_reset_loop
    end : st_reset
    else begin : st_capture
      for (iter_var0 = 0; iter_var0 < NUM_PORTS; iter_var0 = iter_var0 + 1) begin : st_capture_loop
        st_out_flit[iter_var0]  <= xbar_out_flits[iter_var0];
        st_out_valid[iter_var0] <= xbar_out_valid[iter_var0];
        st_out_vc[iter_var0]    <= xbar_out_vc[iter_var0];
      end : st_capture_loop
    end : st_capture
  end : st_register

  // ---------------------------------------------------------------
  // Output assignments
  // ---------------------------------------------------------------
  always_comb begin : out_assign
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_PORTS; iter_var0 = iter_var0 + 1) begin : out_loop
      out_flit[iter_var0]  = st_out_flit[iter_var0];
      out_valid[iter_var0] = st_out_valid[iter_var0];
    end : out_loop
  end : out_assign

endmodule : noc_router
