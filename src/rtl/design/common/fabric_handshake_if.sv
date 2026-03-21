// fabric_handshake_if.sv -- Parameterized valid/ready handshake interface.
//
// Standard data-flow interface used by all fabric modules.  When
// TAG_WIDTH > 0 a sideband tag field is included for temporal routing
// and memory-lane identification.

interface fabric_handshake_if #(
  parameter int unsigned DATA_WIDTH = 32,
  parameter int unsigned TAG_WIDTH  = 0
);

  // ---------------------------------------------------------------
  // Signals
  // ---------------------------------------------------------------

  logic                    valid;
  logic                    ready;
  logic [DATA_WIDTH-1:0]   data;

  // Tag field is conditionally generated.  When TAG_WIDTH == 0 the
  // generate block produces no signals and modports omit the tag.
  generate
    if (TAG_WIDTH > 0) begin : gen_tag
      logic [TAG_WIDTH-1:0] tag;
    end : gen_tag
  endgenerate

  // Transfer indicator -- asserted for exactly one cycle when a
  // payload is consumed by the sink.
  logic transfer;
  assign transfer = valid & ready;

  // ---------------------------------------------------------------
  // Modports
  // ---------------------------------------------------------------

  generate
    if (TAG_WIDTH > 0) begin : gen_modport_tagged
      // Modports are declared inside generate so they can reference
      // the conditionally-generated tag signal.
    end : gen_modport_tagged
  endgenerate

  // Source drives valid, data (and tag when present); reads ready.
  modport source (
    output valid,
    output data,
    input  ready
  );

  // Sink drives ready; reads valid, data (and tag when present).
  modport sink (
    input  valid,
    input  data,
    output ready
  );

endinterface : fabric_handshake_if
