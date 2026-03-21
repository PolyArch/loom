// fabric_handshake_if.sv -- Parameterized valid/ready handshake interface.
//
// Standard data-flow interface used by all fabric modules.  When
// TAG_WIDTH > 0 a sideband tag field carries temporal routing or
// memory-lane identification.  When TAG_WIDTH == 0, a 1-bit dummy
// tag is always present in the modports for port uniformity; the
// source must tie it to 0.

interface fabric_handshake_if #(
  parameter int unsigned DATA_WIDTH = 32,
  parameter int unsigned TAG_WIDTH  = 0
);

  // Effective tag width: at least 1 bit for port uniformity.
  localparam int unsigned TAG_W   = (TAG_WIDTH > 0) ? TAG_WIDTH : 1;
  localparam bit          HAS_TAG = (TAG_WIDTH > 0);

  // ---------------------------------------------------------------
  // Signals
  // ---------------------------------------------------------------

  logic                  valid;
  logic                  ready;
  logic [DATA_WIDTH-1:0] data;
  logic [TAG_W-1:0]      tag;

  // Transfer indicator -- asserted when a payload is consumed.
  logic transfer;
  assign transfer = valid & ready;

  // ---------------------------------------------------------------
  // Modports -- tag is always exposed for port uniformity.
  // ---------------------------------------------------------------

  // Source drives valid, data, tag; reads ready.
  modport source (
    output valid,
    output data,
    output tag,
    input  ready
  );

  // Sink drives ready; reads valid, data, tag.
  modport sink (
    input  valid,
    input  data,
    input  tag,
    output ready
  );

endinterface : fabric_handshake_if
