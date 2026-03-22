// fabric_del_tag.sv -- Strip tag from a tagged value, forward data only.
//
// Pure combinational datapath.  No configuration.  The input tag
// field is discarded; only the data payload is forwarded.

module fabric_del_tag #(
  parameter int unsigned DATA_WIDTH  = 32,
  parameter int unsigned IN_TAG_WIDTH = 4
)(
  input  logic                      clk,
  input  logic                      rst_n,

  // --- Input (tagged value) ---
  input  logic                      in_valid,
  output logic                      in_ready,
  input  logic [DATA_WIDTH-1:0]     in_data,
  input  logic [IN_TAG_WIDTH-1:0]   in_tag,

  // --- Output (untagged value) ---
  output logic                      out_valid,
  input  logic                      out_ready,
  output logic [DATA_WIDTH-1:0]     out_data
);

  // ---------------------------------------------------------------
  // Combinational datapath: passthrough data, discard tag
  // ---------------------------------------------------------------
  assign out_valid = in_valid;
  assign in_ready  = out_ready;
  assign out_data  = in_data;

endmodule : fabric_del_tag
