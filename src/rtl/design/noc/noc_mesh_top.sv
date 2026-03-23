// noc_mesh_top.sv -- Top-level mesh interconnect.
//
// Instantiates a MESH_ROWS x MESH_COLS grid of noc_router instances
// and wires adjacent routers together.  Boundary routers have their
// unused directional ports tied off (valid=0, credit=0).
//
// Each router exposes a LOCAL port that connects to the core injection
// and ejection interfaces.

module noc_mesh_top
  import noc_pkg::*;
#(
  parameter int unsigned MESH_ROWS    = 2,
  parameter int unsigned MESH_COLS    = 2,
  parameter int unsigned DATA_WIDTH   = NOC_DATA_WIDTH_DEFAULT,
  parameter int unsigned NUM_VC       = NOC_NUM_VC_DEFAULT,
  parameter int unsigned BUFFER_DEPTH = NOC_BUFFER_DEPTH_DEFAULT
)(
  input  logic clk,
  input  logic rst_n,

  // Core-facing injection ports (one per router).
  input  logic [flit_width(DATA_WIDTH)-1:0] core_inject_flit  [MESH_ROWS * MESH_COLS],
  input  logic                              core_inject_valid [MESH_ROWS * MESH_COLS],
  output logic                              core_inject_ready [MESH_ROWS * MESH_COLS],

  // Core-facing ejection ports (one per router).
  output logic [flit_width(DATA_WIDTH)-1:0] core_eject_flit   [MESH_ROWS * MESH_COLS],
  output logic                              core_eject_valid  [MESH_ROWS * MESH_COLS],
  input  logic                              core_eject_ready  [MESH_ROWS * MESH_COLS]
);

  // ---------------------------------------------------------------
  // Localparams
  // ---------------------------------------------------------------
  localparam int unsigned FLIT_W    = flit_width(DATA_WIDTH);
  localparam int unsigned NUM_PORTS = NOC_NUM_PORTS;
  localparam int unsigned NUM_ROUTERS = MESH_ROWS * MESH_COLS;

  // Direction indices (matching direction_t encoding).
  localparam int unsigned NORTH = 0;
  localparam int unsigned EAST  = 1;
  localparam int unsigned SOUTH = 2;
  localparam int unsigned WEST  = 3;
  localparam int unsigned LOCAL = 4;

  // ---------------------------------------------------------------
  // Inter-router wires
  // ---------------------------------------------------------------
  // Per-router port signals.
  logic [FLIT_W-1:0] r_in_flit   [NUM_ROUTERS][NUM_PORTS];
  logic               r_in_valid  [NUM_ROUTERS][NUM_PORTS];
  logic               r_in_ready  [NUM_ROUTERS][NUM_PORTS];

  logic [FLIT_W-1:0] r_out_flit  [NUM_ROUTERS][NUM_PORTS];
  logic               r_out_valid [NUM_ROUTERS][NUM_PORTS];
  logic               r_out_ready [NUM_ROUTERS][NUM_PORTS];

  logic [NUM_VC-1:0]  r_credit_out [NUM_ROUTERS][NUM_PORTS];
  logic [NUM_VC-1:0]  r_credit_in  [NUM_ROUTERS][NUM_PORTS];

  // ---------------------------------------------------------------
  // Router instances
  // ---------------------------------------------------------------
  generate
    genvar gv_row, gv_col;
    for (gv_row = 0; gv_row < MESH_ROWS; gv_row = gv_row + 1) begin : gen_row
      for (gv_col = 0; gv_col < MESH_COLS; gv_col = gv_col + 1) begin : gen_col

        localparam int unsigned RID = gv_row * MESH_COLS + gv_col;

        noc_router #(
          .DATA_WIDTH   (DATA_WIDTH),
          .NUM_VC       (NUM_VC),
          .BUFFER_DEPTH (BUFFER_DEPTH),
          .NUM_PORTS    (NUM_PORTS),
          .ROUTER_ROW   (gv_row),
          .ROUTER_COL   (gv_col),
          .MESH_ROWS    (MESH_ROWS),
          .MESH_COLS    (MESH_COLS)
        ) u_router (
          .clk        (clk),
          .rst_n      (rst_n),
          .in_flit    (r_in_flit[RID]),
          .in_valid   (r_in_valid[RID]),
          .in_ready   (r_in_ready[RID]),
          .out_flit   (r_out_flit[RID]),
          .out_valid  (r_out_valid[RID]),
          .out_ready  (r_out_ready[RID]),
          .credit_out (r_credit_out[RID]),
          .credit_in  (r_credit_in[RID])
        );

      end : gen_col
    end : gen_row
  endgenerate

  // ---------------------------------------------------------------
  // Inter-router wiring and boundary tie-off
  // ---------------------------------------------------------------
  generate
    genvar gv_wr, gv_wc;
    for (gv_wr = 0; gv_wr < MESH_ROWS; gv_wr = gv_wr + 1) begin : gen_wire_row
      for (gv_wc = 0; gv_wc < MESH_COLS; gv_wc = gv_wc + 1) begin : gen_wire_col

        localparam int unsigned CUR = gv_wr * MESH_COLS + gv_wc;

        // --- NORTH port ---
        if (gv_wr > 0) begin : gen_north_link
          localparam int unsigned NBR = (gv_wr - 1) * MESH_COLS + gv_wc;
          // This router's NORTH input <- neighbor's SOUTH output.
          assign r_in_flit[CUR][NORTH]   = r_out_flit[NBR][SOUTH];
          assign r_in_valid[CUR][NORTH]   = r_out_valid[NBR][SOUTH];
          assign r_out_ready[CUR][NORTH]  = 1'b1;  // credit-based, always accept
          assign r_credit_in[CUR][NORTH]  = r_credit_out[NBR][SOUTH];
        end : gen_north_link
        else begin : gen_north_tieoff
          assign r_in_flit[CUR][NORTH]   = '0;
          assign r_in_valid[CUR][NORTH]   = 1'b0;
          assign r_out_ready[CUR][NORTH]  = 1'b0;
          assign r_credit_in[CUR][NORTH]  = '0;
        end : gen_north_tieoff

        // --- SOUTH port ---
        if (gv_wr < MESH_ROWS - 1) begin : gen_south_link
          localparam int unsigned NBR = (gv_wr + 1) * MESH_COLS + gv_wc;
          assign r_in_flit[CUR][SOUTH]   = r_out_flit[NBR][NORTH];
          assign r_in_valid[CUR][SOUTH]   = r_out_valid[NBR][NORTH];
          assign r_out_ready[CUR][SOUTH]  = 1'b1;
          assign r_credit_in[CUR][SOUTH]  = r_credit_out[NBR][NORTH];
        end : gen_south_link
        else begin : gen_south_tieoff
          assign r_in_flit[CUR][SOUTH]   = '0;
          assign r_in_valid[CUR][SOUTH]   = 1'b0;
          assign r_out_ready[CUR][SOUTH]  = 1'b0;
          assign r_credit_in[CUR][SOUTH]  = '0;
        end : gen_south_tieoff

        // --- EAST port ---
        if (gv_wc < MESH_COLS - 1) begin : gen_east_link
          localparam int unsigned NBR = gv_wr * MESH_COLS + (gv_wc + 1);
          assign r_in_flit[CUR][EAST]   = r_out_flit[NBR][WEST];
          assign r_in_valid[CUR][EAST]   = r_out_valid[NBR][WEST];
          assign r_out_ready[CUR][EAST]  = 1'b1;
          assign r_credit_in[CUR][EAST]  = r_credit_out[NBR][WEST];
        end : gen_east_link
        else begin : gen_east_tieoff
          assign r_in_flit[CUR][EAST]   = '0;
          assign r_in_valid[CUR][EAST]   = 1'b0;
          assign r_out_ready[CUR][EAST]  = 1'b0;
          assign r_credit_in[CUR][EAST]  = '0;
        end : gen_east_tieoff

        // --- WEST port ---
        if (gv_wc > 0) begin : gen_west_link
          localparam int unsigned NBR = gv_wr * MESH_COLS + (gv_wc - 1);
          assign r_in_flit[CUR][WEST]   = r_out_flit[NBR][EAST];
          assign r_in_valid[CUR][WEST]   = r_out_valid[NBR][EAST];
          assign r_out_ready[CUR][WEST]  = 1'b1;
          assign r_credit_in[CUR][WEST]  = r_credit_out[NBR][EAST];
        end : gen_west_link
        else begin : gen_west_tieoff
          assign r_in_flit[CUR][WEST]   = '0;
          assign r_in_valid[CUR][WEST]   = 1'b0;
          assign r_out_ready[CUR][WEST]  = 1'b0;
          assign r_credit_in[CUR][WEST]  = '0;
        end : gen_west_tieoff

        // --- LOCAL port: connect to core interfaces ---
        assign r_in_flit[CUR][LOCAL]   = core_inject_flit[CUR];
        assign r_in_valid[CUR][LOCAL]   = core_inject_valid[CUR];
        assign core_inject_ready[CUR]   = r_in_ready[CUR][LOCAL];

        assign core_eject_flit[CUR]     = r_out_flit[CUR][LOCAL];
        assign core_eject_valid[CUR]    = r_out_valid[CUR][LOCAL];
        assign r_out_ready[CUR][LOCAL]  = core_eject_ready[CUR];
        assign r_credit_in[CUR][LOCAL]  = '0;  // Local port: no credit from core

      end : gen_wire_col
    end : gen_wire_row
  endgenerate

endmodule : noc_mesh_top
