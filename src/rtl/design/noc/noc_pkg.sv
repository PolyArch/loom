// noc_pkg.sv -- Shared package for NoC RTL infrastructure.
//
// Provides common types, constants, and parameter defaults used across
// all NoC design modules.  Flit format, routing direction enumerations,
// and header/flit structs are defined here.

package noc_pkg;

  // ---------------------------------------------------------------
  // Default parameters
  // ---------------------------------------------------------------
  localparam int unsigned NOC_DATA_WIDTH_DEFAULT  = 32;
  localparam int unsigned NOC_NUM_VC_DEFAULT       = 2;
  localparam int unsigned NOC_BUFFER_DEPTH_DEFAULT = 4;
  localparam int unsigned NOC_NUM_PORTS            = 5;  // N, E, S, W, Local

  // Router-ID width: enough bits for up to 16x16 mesh (256 nodes).
  localparam int unsigned NOC_ID_WIDTH             = 8;

  // ---------------------------------------------------------------
  // Flit type enumeration
  // ---------------------------------------------------------------
  typedef enum logic [1:0] {
    FLIT_HEAD   = 2'b00,
    FLIT_BODY   = 2'b01,
    FLIT_TAIL   = 2'b10,
    FLIT_SINGLE = 2'b11
  } flit_type_t;

  // ---------------------------------------------------------------
  // Direction enumeration (port index)
  // ---------------------------------------------------------------
  typedef enum logic [2:0] {
    DIR_NORTH = 3'd0,
    DIR_EAST  = 3'd1,
    DIR_SOUTH = 3'd2,
    DIR_WEST  = 3'd3,
    DIR_LOCAL = 3'd4
  } direction_t;

  // ---------------------------------------------------------------
  // Flit header -- packed into the upper bits of a flit
  // ---------------------------------------------------------------
  // Total header width = 2 (type) + ID_WIDTH (src) + ID_WIDTH (dst)
  //                    + clog2(NUM_VC) (vc_id)
  // For simplicity, vc_id is 1 bit (supports up to 2 VCs; widen if
  // NUM_VC > 2 is needed).

  localparam int unsigned NOC_VC_ID_WIDTH = 1;

  localparam int unsigned NOC_HEADER_WIDTH = 2 + NOC_ID_WIDTH
                                             + NOC_ID_WIDTH
                                             + NOC_VC_ID_WIDTH;

  typedef struct packed {
    flit_type_t                  flit_type;  // [MSB : MSB-1]
    logic [NOC_ID_WIDTH-1:0]     src_id;
    logic [NOC_ID_WIDTH-1:0]     dst_id;
    logic [NOC_VC_ID_WIDTH-1:0]  vc_id;
  } flit_header_t;

  // ---------------------------------------------------------------
  // Utility: compute flit width from data width
  // ---------------------------------------------------------------
  // Full flit = header + payload.  The payload occupies DATA_WIDTH
  // bits.  Total flit width = NOC_HEADER_WIDTH + DATA_WIDTH.
  function automatic int unsigned flit_width(input int unsigned data_w);
    return NOC_HEADER_WIDTH + data_w;
  endfunction : flit_width

endpackage : noc_pkg
