// noc_xy_route.sv -- XY dimension-ordered route computation.
//
// Combinational module that computes the output direction for a flit
// based on its destination ID and the current router position in the
// mesh.  Uses X-first (column-first) routing: if the destination
// column differs, route EAST or WEST; otherwise, if the destination
// row differs, route NORTH or SOUTH; otherwise, route to LOCAL.
//
// The destination ID is decoded as:
//   dst_row = dst_id / MESH_COLS
//   dst_col = dst_id % MESH_COLS

module noc_xy_route
  import noc_pkg::*;
#(
  parameter int unsigned ROUTER_ROW = 0,
  parameter int unsigned ROUTER_COL = 0,
  parameter int unsigned MESH_COLS  = 2
)(
  input  logic [NOC_ID_WIDTH-1:0] dst_id,
  output direction_t              out_dir
);

  // ---------------------------------------------------------------
  // Decode destination coordinates
  // ---------------------------------------------------------------
  logic [NOC_ID_WIDTH-1:0] dst_row;
  logic [NOC_ID_WIDTH-1:0] dst_col;

  assign dst_row = dst_id / NOC_ID_WIDTH'(MESH_COLS);
  assign dst_col = dst_id % NOC_ID_WIDTH'(MESH_COLS);

  // ---------------------------------------------------------------
  // XY routing decision (combinational)
  // ---------------------------------------------------------------
  /* verilator lint_off UNSIGNED */
  always_comb begin : xy_route_compute
    if (dst_col < NOC_ID_WIDTH'(ROUTER_COL)) begin : route_west
      out_dir = DIR_WEST;
    end : route_west
    else if (dst_col > NOC_ID_WIDTH'(ROUTER_COL)) begin : route_east
      out_dir = DIR_EAST;
    end : route_east
    else if (dst_row < NOC_ID_WIDTH'(ROUTER_ROW)) begin : route_north
      out_dir = DIR_NORTH;
    end : route_north
    else if (dst_row > NOC_ID_WIDTH'(ROUTER_ROW)) begin : route_south
      out_dir = DIR_SOUTH;
    end : route_south
    else begin : route_local
      out_dir = DIR_LOCAL;
    end : route_local
  end : xy_route_compute
  /* verilator lint_on UNSIGNED */

endmodule : noc_xy_route
