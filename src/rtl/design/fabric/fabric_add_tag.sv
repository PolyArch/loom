// fabric_add_tag.sv -- Attach a constant tag to an untagged value.
//
// Pure combinational datapath.  The tag value is loaded via a
// word-serial config bus (one 32-bit word; low TAG_WIDTH bits used).
// Handshake: valid/ready passthrough from input to output.

module fabric_add_tag #(
  parameter int unsigned DATA_WIDTH = 32,
  parameter int unsigned TAG_WIDTH  = 4
)(
  input  logic                    clk,
  input  logic                    rst_n,

  // --- Config port (word-serial, 1 word) ---
  input  logic                    cfg_valid,
  input  logic [31:0]             cfg_wdata,
  output logic                    cfg_ready,

  // --- Input (untagged value) ---
  input  logic                    in_valid,
  output logic                    in_ready,
  input  logic [DATA_WIDTH-1:0]   in_data,

  // --- Output (tagged value) ---
  output logic                    out_valid,
  input  logic                    out_ready,
  output logic [DATA_WIDTH-1:0]   out_data,
  output logic [TAG_WIDTH-1:0]    out_tag
);

  // ---------------------------------------------------------------
  // Config register: tag value
  // ---------------------------------------------------------------
  logic [TAG_WIDTH-1:0] cfg_tag;

  // Config loading: accept one word, store low TAG_WIDTH bits.
  // cfg_ready is always high -- single-word config, no backpressure.
  assign cfg_ready = 1'b1;

  always_ff @(posedge clk) begin : cfg_load
    if (!rst_n) begin : cfg_reset
      cfg_tag <= '0;
    end : cfg_reset
    else begin : cfg_update
      if (cfg_valid && cfg_ready) begin : cfg_capture
        cfg_tag <= cfg_wdata[TAG_WIDTH-1:0];
      end : cfg_capture
    end : cfg_update
  end : cfg_load

  // ---------------------------------------------------------------
  // Combinational datapath: passthrough data, attach constant tag
  // ---------------------------------------------------------------
  assign out_valid = in_valid;
  assign in_ready  = out_ready;
  assign out_data  = in_data;
  assign out_tag   = cfg_tag;

endmodule : fabric_add_tag
