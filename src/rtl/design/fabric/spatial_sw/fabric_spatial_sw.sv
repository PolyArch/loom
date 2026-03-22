// fabric_spatial_sw.sv -- Top-level spatial switch wrapper.
//
// Configurable crossbar routing switch.  Routing decisions are based
// solely on the configured route_table bitmap, never on tag values.
// Tags, when present, travel as part of the payload.
//
// Two variants are selected at elaboration time:
//   - DECOMPOSABLE_BITS == 0: non-decomposable crossbar (fabric_spatial_sw_core)
//   - DECOMPOSABLE_BITS >  0: decomposable sub-lane crossbar (fabric_spatial_sw_decomp)
//
// Config bitstream layout (matching spec-fabric-config_mem.md):
//   route_bits = popcount(CONNECTIVITY)
//   discard_bits = NUM_IN
//   total config bits = route_bits + discard_bits
//   Bit order: route bitmap (output-major, input-major within each output,
//              only connected positions), then per-input discard bits.

module fabric_spatial_sw
  import fabric_pkg::*;
#(
  parameter int unsigned NUM_IN            = 2,
  parameter int unsigned NUM_OUT           = 2,
  parameter int unsigned DATA_WIDTH        = 32,
  parameter int unsigned TAG_WIDTH         = 0,
  // Connectivity matrix: packed bit array [NUM_OUT][NUM_IN].
  // Bit [out_idx * NUM_IN + in_idx] == 1 means input in_idx may drive output out_idx.
  // Default: full connectivity.
  parameter bit [NUM_OUT*NUM_IN-1:0] CONNECTIVITY = {NUM_OUT*NUM_IN{1'b1}},
  // Sub-lane decomposition granularity in bits.  0 means non-decomposable.
  parameter int unsigned DECOMPOSABLE_BITS = 0
)(
  input  logic                    clk,
  input  logic                    rst_n,

  // --- Per-input handshake ---
  input  logic [NUM_IN-1:0]       in_valid,
  output logic [NUM_IN-1:0]       in_ready,
  input  logic [DATA_WIDTH-1:0]   in_data  [NUM_IN],
  // Tag port present for interface uniformity; unused by decomposable variant.
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [TAG_WIDTH > 0 ? TAG_WIDTH-1 : 0 : 0]
                                  in_tag   [NUM_IN],
  /* verilator lint_on UNUSEDSIGNAL */

  // --- Per-output handshake ---
  output logic [NUM_OUT-1:0]      out_valid,
  input  logic [NUM_OUT-1:0]      out_ready,
  output logic [DATA_WIDTH-1:0]   out_data [NUM_OUT],
  output logic [TAG_WIDTH > 0 ? TAG_WIDTH-1 : 0 : 0]
                                  out_tag  [NUM_OUT],

  // --- Config port (word-serial) ---
  input  logic                    cfg_valid,
  input  logic [31:0]             cfg_wdata,
  output logic                    cfg_ready
);

  // ---------------------------------------------------------------
  // Variant selection
  // ---------------------------------------------------------------
  generate
    if (DECOMPOSABLE_BITS == 0) begin : gen_core

      fabric_spatial_sw_core #(
        .NUM_IN       (NUM_IN),
        .NUM_OUT      (NUM_OUT),
        .DATA_WIDTH   (DATA_WIDTH),
        .TAG_WIDTH    (TAG_WIDTH),
        .CONNECTIVITY (CONNECTIVITY)
      ) u_core (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (in_valid),
        .in_ready  (in_ready),
        .in_data   (in_data),
        .in_tag    (in_tag),
        .out_valid (out_valid),
        .out_ready (out_ready),
        .out_data  (out_data),
        .out_tag   (out_tag),
        .cfg_valid (cfg_valid),
        .cfg_wdata (cfg_wdata),
        .cfg_ready (cfg_ready)
      );

    end : gen_core
    else begin : gen_decomp

      fabric_spatial_sw_decomp #(
        .NUM_IN           (NUM_IN),
        .NUM_OUT          (NUM_OUT),
        .DATA_WIDTH       (DATA_WIDTH),
        .CONNECTIVITY     (CONNECTIVITY),
        .DECOMPOSABLE_BITS(DECOMPOSABLE_BITS)
      ) u_decomp (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (in_valid),
        .in_ready  (in_ready),
        .in_data   (in_data),
        .out_valid (out_valid),
        .out_ready (out_ready),
        .out_data  (out_data),
        .cfg_valid (cfg_valid),
        .cfg_wdata (cfg_wdata),
        .cfg_ready (cfg_ready)
      );

      // Decomposable switches do not support tags.
      // Drive tag outputs to zero.
      genvar gzt;
      for (gzt = 0; gzt < NUM_OUT; gzt = gzt + 1) begin : gen_zero_tag
        assign out_tag[gzt] = '0;
      end : gen_zero_tag

    end : gen_decomp
  endgenerate

endmodule : fabric_spatial_sw
