// noc_input_port.sv -- Per-port input buffer and VC management.
//
// Each input port contains NUM_VC independent FIFO buffers (one per
// virtual channel).  Incoming flits are steered to the correct VC
// buffer based on the vc_id field in the flit header.
//
// Route computation is performed on HEAD and SINGLE flits using the
// noc_xy_route module.  The computed direction is latched and held
// for the duration of a multi-flit message (wormhole switching).
//
// Credit-based flow control: a credit pulse is sent upstream when a
// buffer slot is freed (i.e., when a flit is granted and dequeued).
// The flit_ready output indicates whether the port can accept a new
// flit (at least one VC buffer has space).

module noc_input_port
  import noc_pkg::*;
#(
  parameter int unsigned DATA_WIDTH   = NOC_DATA_WIDTH_DEFAULT,
  parameter int unsigned NUM_VC       = NOC_NUM_VC_DEFAULT,
  parameter int unsigned BUFFER_DEPTH = NOC_BUFFER_DEPTH_DEFAULT,
  parameter int unsigned ROUTER_ROW   = 0,
  parameter int unsigned ROUTER_COL   = 0,
  parameter int unsigned MESH_COLS    = 2
)(
  input  logic clk,
  input  logic rst_n,

  // Flit input from upstream.
  input  logic [flit_width(DATA_WIDTH)-1:0] flit_in,
  input  logic                              flit_valid,
  output logic                              flit_ready,

  // Route-compute result to arbiter (for the selected VC).
  output logic                              rc_valid,
  output direction_t                        rc_out_dir,
  output logic [NOC_VC_ID_WIDTH-1:0]        rc_vc,
  output logic [flit_width(DATA_WIDTH)-1:0] rc_flit,

  // Grant from arbiter -- dequeue the selected VC buffer.
  input  logic                              grant,

  // Credit pulse to upstream (one bit per VC).
  output logic [NUM_VC-1:0]                 credit_out
);

  // ---------------------------------------------------------------
  // Localparams
  // ---------------------------------------------------------------
  localparam int unsigned FLIT_W = flit_width(DATA_WIDTH);

  // ---------------------------------------------------------------
  // VC-ID extraction from incoming flit header
  // ---------------------------------------------------------------
  // In the packed flit layout {flit_type, src_id, dst_id, vc_id, payload},
  // vc_id occupies bits [DATA_WIDTH +: NOC_VC_ID_WIDTH].
  logic [NOC_VC_ID_WIDTH-1:0] in_vc_id;
  assign in_vc_id = flit_in[DATA_WIDTH +: NOC_VC_ID_WIDTH];

  // ---------------------------------------------------------------
  // Per-VC FIFO storage
  // ---------------------------------------------------------------
  // FIFO control signals per VC.
  logic [NUM_VC-1:0]  vc_push;
  logic [NUM_VC-1:0]  vc_pop;
  logic [FLIT_W-1:0]  vc_dout   [NUM_VC];
  logic [NUM_VC-1:0]  vc_full;
  logic [NUM_VC-1:0]  vc_empty;

  generate
    genvar gv_vc;
    for (gv_vc = 0; gv_vc < NUM_VC; gv_vc = gv_vc + 1) begin : gen_vc_fifo
      fabric_fifo_mem #(
        .DEPTH      (BUFFER_DEPTH),
        .DATA_WIDTH (FLIT_W)
      ) u_fifo (
        .clk   (clk),
        .rst_n (rst_n),
        .push  (vc_push[gv_vc]),
        .din   (flit_in),
        .pop   (vc_pop[gv_vc]),
        .dout  (vc_dout[gv_vc]),
        .full  (vc_full[gv_vc]),
        .empty (vc_empty[gv_vc]),
        /* verilator lint_off PINCONNECTEMPTY */
        .count ()
        /* verilator lint_on PINCONNECTEMPTY */
      );
    end : gen_vc_fifo
  endgenerate

  // ---------------------------------------------------------------
  // Input demux: steer incoming flit to the correct VC FIFO
  // ---------------------------------------------------------------
  always_comb begin : input_demux
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_VC; iter_var0 = iter_var0 + 1) begin : vc_push_gen
      if (flit_valid && (in_vc_id == NOC_VC_ID_WIDTH'(iter_var0))
          && !vc_full[iter_var0]) begin : vc_push_match
        vc_push[iter_var0] = 1'b1;
      end : vc_push_match
      else begin : vc_push_no_match
        vc_push[iter_var0] = 1'b0;
      end : vc_push_no_match
    end : vc_push_gen
  end : input_demux

  // flit_ready: accept if the target VC has space.
  always_comb begin : ready_gen
    integer iter_var0;
    flit_ready = 1'b0;
    for (iter_var0 = 0; iter_var0 < NUM_VC; iter_var0 = iter_var0 + 1) begin : ready_scan
      if (in_vc_id == NOC_VC_ID_WIDTH'(iter_var0)) begin : ready_match
        flit_ready = ~vc_full[iter_var0];
      end : ready_match
    end : ready_scan
  end : ready_gen

  // ---------------------------------------------------------------
  // Per-VC route compute: latch direction on HEAD/SINGLE flits
  // ---------------------------------------------------------------
  direction_t vc_route_dir [NUM_VC];
  logic [NUM_VC-1:0] vc_route_valid;

  // Combinational XY route for the head at the front of each VC.
  direction_t vc_xy_dir [NUM_VC];

  generate
    genvar gv_rc;
    for (gv_rc = 0; gv_rc < NUM_VC; gv_rc = gv_rc + 1) begin : gen_vc_rc

      // Extract dst_id from the FIFO head's header.
      // In packed layout: {flit_type[1:0], src_id[ID_W-1:0], dst_id[ID_W-1:0], vc_id}
      // dst_id starts at bit position (DATA_WIDTH + NOC_VC_ID_WIDTH).
      logic [NOC_ID_WIDTH-1:0] vc_head_dst_id;
      assign vc_head_dst_id = vc_dout[gv_rc][DATA_WIDTH + NOC_VC_ID_WIDTH +: NOC_ID_WIDTH];

      noc_xy_route #(
        .ROUTER_ROW (ROUTER_ROW),
        .ROUTER_COL (ROUTER_COL),
        .MESH_COLS  (MESH_COLS)
      ) u_xy_route (
        .dst_id  (vc_head_dst_id),
        .out_dir (vc_xy_dir[gv_rc])
      );

    end : gen_vc_rc
  endgenerate

  // Route-valid tracking: a VC has a valid route when it is non-empty
  // and the head flit has been route-computed.  For HEAD/SINGLE flits,
  // the route is valid immediately from the combinational XY module.
  // For BODY/TAIL flits, the route was latched on the preceding HEAD.
  always_ff @(posedge clk or negedge rst_n) begin : route_latch
    integer iter_var0;
    if (!rst_n) begin : route_latch_reset
      for (iter_var0 = 0; iter_var0 < NUM_VC; iter_var0 = iter_var0 + 1) begin : route_reset_loop
        vc_route_dir[iter_var0]  <= DIR_LOCAL;
        vc_route_valid[iter_var0] <= 1'b0;
      end : route_reset_loop
    end : route_latch_reset
    else begin : route_latch_update
      for (iter_var0 = 0; iter_var0 < NUM_VC; iter_var0 = iter_var0 + 1) begin : route_update_loop
        if (!vc_empty[iter_var0]) begin : route_nonempty
          // Extract flit type from FIFO head.
          automatic flit_type_t head_type;
          head_type = flit_type_t'(vc_dout[iter_var0][FLIT_W-1 -: 2]);

          if (head_type == FLIT_HEAD || head_type == FLIT_SINGLE) begin : route_head
            vc_route_dir[iter_var0]   <= vc_xy_dir[iter_var0];
            vc_route_valid[iter_var0] <= 1'b1;
          end : route_head
        end : route_nonempty

        // Clear route valid when the tail or single flit is dequeued.
        if (vc_pop[iter_var0]) begin : route_pop_check
          automatic flit_type_t pop_type;
          pop_type = flit_type_t'(vc_dout[iter_var0][FLIT_W-1 -: 2]);
          if (pop_type == FLIT_TAIL || pop_type == FLIT_SINGLE) begin : route_clear
            vc_route_valid[iter_var0] <= 1'b0;
          end : route_clear
        end : route_pop_check
      end : route_update_loop
    end : route_latch_update
  end : route_latch

  // ---------------------------------------------------------------
  // VC selection: round-robin among VCs with valid routes
  // ---------------------------------------------------------------
  logic [$clog2(NUM_VC > 1 ? NUM_VC : 2)-1:0] vc_rr_ptr;
  logic [$clog2(NUM_VC > 1 ? NUM_VC : 2)-1:0] selected_vc;
  logic                                        any_vc_ready;

  // Use combinational route for HEAD/SINGLE (available immediately),
  // latched route for BODY/TAIL.
  direction_t vc_effective_dir [NUM_VC];

  always_comb begin : vc_eff_dir_compute
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_VC; iter_var0 = iter_var0 + 1) begin : vc_eff_loop
      automatic flit_type_t head_type_c;
      if (!vc_empty[iter_var0]) begin : vc_eff_nonempty
        head_type_c = flit_type_t'(vc_dout[iter_var0][FLIT_W-1 -: 2]);
        if (head_type_c == FLIT_HEAD || head_type_c == FLIT_SINGLE) begin : vc_eff_head
          vc_effective_dir[iter_var0] = vc_xy_dir[iter_var0];
        end : vc_eff_head
        else begin : vc_eff_body
          vc_effective_dir[iter_var0] = vc_route_dir[iter_var0];
        end : vc_eff_body
      end : vc_eff_nonempty
      else begin : vc_eff_empty
        vc_effective_dir[iter_var0] = DIR_LOCAL;
      end : vc_eff_empty
    end : vc_eff_loop
  end : vc_eff_dir_compute

  // VC eligibility: non-empty and (has latched route OR is HEAD/SINGLE).
  logic [NUM_VC-1:0] vc_eligible;

  always_comb begin : vc_eligible_compute
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_VC; iter_var0 = iter_var0 + 1) begin : vc_elig_loop
      if (!vc_empty[iter_var0]) begin : vc_elig_nonempty
        automatic flit_type_t head_type_e;
        head_type_e = flit_type_t'(vc_dout[iter_var0][FLIT_W-1 -: 2]);
        if (head_type_e == FLIT_HEAD || head_type_e == FLIT_SINGLE) begin : vc_elig_head
          vc_eligible[iter_var0] = 1'b1;
        end : vc_elig_head
        else begin : vc_elig_body
          vc_eligible[iter_var0] = vc_route_valid[iter_var0];
        end : vc_elig_body
      end : vc_elig_nonempty
      else begin : vc_elig_empty
        vc_eligible[iter_var0] = 1'b0;
      end : vc_elig_empty
    end : vc_elig_loop
  end : vc_eligible_compute

  // Round-robin selection among eligible VCs.
  localparam int unsigned VC_IDX_W = $clog2(NUM_VC > 1 ? NUM_VC : 2);

  always_comb begin : vc_select
    integer iter_var0;
    any_vc_ready = 1'b0;
    selected_vc  = '0;

    for (iter_var0 = 0; iter_var0 < NUM_VC; iter_var0 = iter_var0 + 1) begin : vc_scan
      if (!any_vc_ready) begin : vc_scan_check
        automatic logic [VC_IDX_W-1:0] probe_vc;
        probe_vc = VC_IDX_W'((int'(vc_rr_ptr) + iter_var0) % NUM_VC);
        if (vc_eligible[probe_vc]) begin : vc_scan_hit
          selected_vc  = probe_vc;
          any_vc_ready = 1'b1;
        end : vc_scan_hit
      end : vc_scan_check
    end : vc_scan
  end : vc_select

  // Round-robin pointer update.
  always_ff @(posedge clk or negedge rst_n) begin : vc_rr_update
    if (!rst_n) begin : vc_rr_reset
      vc_rr_ptr <= '0;
    end : vc_rr_reset
    else if (grant && any_vc_ready) begin : vc_rr_advance
      if (selected_vc == VC_IDX_W'(NUM_VC - 1)) begin : vc_rr_wrap
        vc_rr_ptr <= '0;
      end : vc_rr_wrap
      else begin : vc_rr_incr
        vc_rr_ptr <= selected_vc + 1'b1;
      end : vc_rr_incr
    end : vc_rr_advance
  end : vc_rr_update

  // ---------------------------------------------------------------
  // Output to arbiter
  // ---------------------------------------------------------------
  assign rc_valid   = any_vc_ready;
  assign rc_out_dir = vc_effective_dir[selected_vc];
  assign rc_vc      = selected_vc[NOC_VC_ID_WIDTH-1:0];
  assign rc_flit    = vc_dout[selected_vc];

  // ---------------------------------------------------------------
  // Pop: dequeue from the granted VC
  // ---------------------------------------------------------------
  always_comb begin : pop_gen
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_VC; iter_var0 = iter_var0 + 1) begin : pop_loop
      vc_pop[iter_var0] = grant && any_vc_ready
                          && (selected_vc == VC_IDX_W'(iter_var0));
    end : pop_loop
  end : pop_gen

  // ---------------------------------------------------------------
  // Credit generation: pulse when a buffer slot is freed
  // ---------------------------------------------------------------
  always_comb begin : credit_gen
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_VC; iter_var0 = iter_var0 + 1) begin : credit_loop
      credit_out[iter_var0] = vc_pop[iter_var0];
    end : credit_loop
  end : credit_gen

endmodule : noc_input_port
