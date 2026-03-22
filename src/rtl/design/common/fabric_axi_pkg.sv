// fabric_axi_pkg.sv -- AXI4 Memory-Mapped constants and enums.
//
// Provides AXI4 response codes, burst types, and helper functions.
// Width-parameterized AXI channel signals are defined as flat ports
// in each module (not as packed structs from this package), because
// SystemVerilog packages cannot have parameterized typedefs.
//
// Modules that need AXI ports should declare signals like:
//   input  logic [ADDR_WIDTH-1:0] awaddr,
//   input  logic [DATA_WIDTH-1:0] wdata,
// and reference constants from this package (e.g., fabric_axi_pkg::AXI_RESP_OKAY).

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
  // AXI4 Field Widths (protocol-fixed, width-independent)
  // ---------------------------------------------------------------

  localparam int unsigned AXI_LEN_WIDTH  = 8;  // AxLEN (burst length)
  localparam int unsigned AXI_SIZE_WIDTH = 3;  // AxSIZE
  localparam int unsigned AXI_RESP_WIDTH = 2;  // xRESP

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

  // ---------------------------------------------------------------
  // Helper: Compute AxSIZE from element size log2 (LOOM convention)
  // ---------------------------------------------------------------
  function automatic logic [AXI_SIZE_WIDTH-1:0] axsize_from_log2(
    input int unsigned elem_size_log2
  );
    return elem_size_log2[AXI_SIZE_WIDTH-1:0];
  endfunction : axsize_from_log2

endpackage : fabric_axi_pkg
