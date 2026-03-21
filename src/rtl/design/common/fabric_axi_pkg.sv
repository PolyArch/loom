// fabric_axi_pkg.sv -- AXI4 Memory-Mapped type definitions.
//
// Provides parameterized channel structs for AXI4 AW, W, B, AR, R
// channels, standard response codes, and helper types.  Used by
// fabric memory and extmemory modules that expose AXI-MM ports.

package fabric_axi_pkg;

  // ---------------------------------------------------------------
  // AXI Response Codes (AXI4 spec Table A3-4)
  // ---------------------------------------------------------------

  typedef enum logic [1:0] {
    AXI_RESP_OKAY   = 2'b00,
    AXI_RESP_EXOKAY = 2'b01,
    AXI_RESP_SLVERR = 2'b10,
    AXI_RESP_DECERR = 2'b11
  } axi_resp_t;

  // ---------------------------------------------------------------
  // AXI Burst Type (AXI4 spec Table A3-3)
  // ---------------------------------------------------------------

  typedef enum logic [1:0] {
    AXI_BURST_FIXED = 2'b00,
    AXI_BURST_INCR  = 2'b01,
    AXI_BURST_WRAP  = 2'b10
  } axi_burst_t;

  // ---------------------------------------------------------------
  // AXI4 Channel Structs
  //
  // These are parameterized by width constants that must be provided
  // at instantiation.  SystemVerilog does not allow parameterized
  // typedefs in packages, so we define structs with maximum widths
  // and use the relevant bit ranges.  Alternatively, modules that
  // instantiate AXI ports declare local typedefs.
  //
  // For convenience, we provide structs at common widths and a set
  // of localparam constants.
  // ---------------------------------------------------------------

  // Common default parameters.
  localparam int unsigned AXI_ADDR_WIDTH_DEFAULT = 32;
  localparam int unsigned AXI_DATA_WIDTH_DEFAULT = 64;
  localparam int unsigned AXI_ID_WIDTH_DEFAULT   = 8;
  localparam int unsigned AXI_LEN_WIDTH          = 8;   // AXI4 burst length
  localparam int unsigned AXI_SIZE_WIDTH         = 3;   // AxSIZE
  localparam int unsigned AXI_STRB_WIDTH_DEFAULT = AXI_DATA_WIDTH_DEFAULT / 8;

  // ----- Write Address Channel (AW) -----
  typedef struct packed {
    logic [AXI_ID_WIDTH_DEFAULT-1:0]   awid;
    logic [AXI_ADDR_WIDTH_DEFAULT-1:0] awaddr;
    logic [AXI_LEN_WIDTH-1:0]         awlen;
    logic [AXI_SIZE_WIDTH-1:0]        awsize;
    axi_burst_t                       awburst;
  } axi_aw_t;

  // ----- Write Data Channel (W) -----
  typedef struct packed {
    logic [AXI_DATA_WIDTH_DEFAULT-1:0] wdata;
    logic [AXI_STRB_WIDTH_DEFAULT-1:0] wstrb;
    logic                              wlast;
  } axi_w_t;

  // ----- Write Response Channel (B) -----
  typedef struct packed {
    logic [AXI_ID_WIDTH_DEFAULT-1:0] bid;
    axi_resp_t                       bresp;
  } axi_b_t;

  // ----- Read Address Channel (AR) -----
  typedef struct packed {
    logic [AXI_ID_WIDTH_DEFAULT-1:0]   arid;
    logic [AXI_ADDR_WIDTH_DEFAULT-1:0] araddr;
    logic [AXI_LEN_WIDTH-1:0]         arlen;
    logic [AXI_SIZE_WIDTH-1:0]        arsize;
    axi_burst_t                       arburst;
  } axi_ar_t;

  // ----- Read Data Channel (R) -----
  typedef struct packed {
    logic [AXI_ID_WIDTH_DEFAULT-1:0]   rid;
    logic [AXI_DATA_WIDTH_DEFAULT-1:0] rdata;
    axi_resp_t                         rresp;
    logic                              rlast;
  } axi_r_t;

  // ---------------------------------------------------------------
  // Helper: Compute AXI strobe width from data width
  // ---------------------------------------------------------------
  function automatic int unsigned axi_strb_width(input int unsigned data_width);
    return data_width / 8;
  endfunction : axi_strb_width

  // ---------------------------------------------------------------
  // Helper: Compute AxSIZE from byte count (must be power of 2)
  // ---------------------------------------------------------------
  function automatic logic [AXI_SIZE_WIDTH-1:0] axsize_from_bytes(
    input int unsigned num_bytes
  );
    logic [AXI_SIZE_WIDTH-1:0] result;
    result = '0;
    case (num_bytes)
      1:    result = 3'b000;
      2:    result = 3'b001;
      4:    result = 3'b010;
      8:    result = 3'b011;
      16:   result = 3'b100;
      32:   result = 3'b101;
      64:   result = 3'b110;
      128:  result = 3'b111;
      default: result = 3'b010; // default to 4 bytes
    endcase
    return result;
  endfunction : axsize_from_bytes

endpackage : fabric_axi_pkg
