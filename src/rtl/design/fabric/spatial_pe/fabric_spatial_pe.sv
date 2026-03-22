// fabric_spatial_pe.sv -- Top-level spatial processing element container.
//
// A spatial PE holds multiple function_unit instances, of which at most
// one is active at runtime, selected by the configured opcode.  Input
// muxes route PE external inputs to FU inputs; output demuxes route FU
// outputs to PE external outputs.
//
// Config layout (low-to-high, matching spec-fabric-config_mem.md):
//   enable         : 1 bit
//   opcode         : clog2(NUM_FU) bits
//   input_mux[i]   : (clog2(NUM_IN) + 2) bits per entry, MAX_FU_IN entries
//   output_demux[j]: (clog2(NUM_OUT) + 2) bits per entry, MAX_FU_OUT entries
//   fu_config      : MAX_FU_CFG_BITS bits
//
// The total config bit count determines the number of 32-bit config
// words consumed from the config bus.
//
// FU bodies are NOT instantiated here.  The SVGen C++ library generates
// a PE wrapper module that instantiates concrete FU bodies and connects
// them through the fu_slot gating wrappers.  This pre-written container
// provides the mux bank, demux bank, config register unpacking, and
// per-FU gating infrastructure.

module fabric_spatial_pe
  import fabric_pkg::*;
#(
  parameter int unsigned NUM_IN          = 4,
  parameter int unsigned NUM_OUT         = 4,
  parameter int unsigned DATA_WIDTH      = 32,
  parameter int unsigned NUM_FU          = 2,
  parameter int unsigned MAX_FU_IN       = 2,
  parameter int unsigned MAX_FU_OUT      = 1,
  parameter int unsigned MAX_FU_CFG_BITS = 0
)(
  input  logic                    clk,
  input  logic                    rst_n,

  // --- Per-input handshake (PE exterior) ---
  input  logic [NUM_IN-1:0]       in_valid,
  output logic [NUM_IN-1:0]       in_ready,
  input  logic [DATA_WIDTH-1:0]   in_data  [NUM_IN],

  // --- Per-output handshake (PE exterior) ---
  output logic [NUM_OUT-1:0]      out_valid,
  input  logic [NUM_OUT-1:0]      out_ready,
  output logic [DATA_WIDTH-1:0]   out_data [NUM_OUT],

  // --- Config port (word-serial) ---
  input  logic                    cfg_valid,
  input  logic [31:0]             cfg_wdata,
  output logic                    cfg_ready,

  // --- FU body interface (active FU's gated handshake) ---
  // These arrays are sized to MAX_FU_IN/MAX_FU_OUT for the single
  // active FU.  The generated PE wrapper connects these to the
  // selected FU body through fu_slot instances.
  output logic [MAX_FU_IN-1:0]    fu_in_valid,
  input  logic [MAX_FU_IN-1:0]    fu_in_ready,
  output logic [DATA_WIDTH-1:0]   fu_in_data   [MAX_FU_IN],

  input  logic [MAX_FU_OUT-1:0]   fu_out_valid,
  output logic [MAX_FU_OUT-1:0]   fu_out_ready,
  input  logic [DATA_WIDTH-1:0]   fu_out_data  [MAX_FU_OUT],

  // --- FU config bits (shared region, active FU interprets) ---
  output logic [MAX_FU_CFG_BITS > 0 ? MAX_FU_CFG_BITS-1 : 0 : 0]
                                  fu_cfg_bits,

  // --- PE enable and opcode (exposed for fu_slot gating) ---
  output logic                    pe_enable,
  output logic [clog2_min1(NUM_FU)-1:0]
                                  pe_opcode
);

  // ---------------------------------------------------------------
  // Derived parameters
  // ---------------------------------------------------------------

  // Opcode width: clog2(NUM_FU).  0 when NUM_FU <= 1.
  localparam int unsigned OPCODE_WIDTH = clog2(NUM_FU);
  // Actual opcode register width: at least 1 bit for signal legality.
  localparam int unsigned OPCODE_REG_W = clog2_min1(NUM_FU);

  // Input mux sel width: clog2(NUM_IN).
  localparam int unsigned IN_MUX_SEL_W = clog2(NUM_IN);
  // Each input mux field: sel + discard + disconnect.
  localparam int unsigned IN_MUX_FIELD_W = IN_MUX_SEL_W + 2;

  // Output demux sel width: clog2(NUM_OUT).
  localparam int unsigned OUT_DEMUX_SEL_W = clog2(NUM_OUT);
  // Each output demux field: sel + discard + disconnect.
  localparam int unsigned OUT_DEMUX_FIELD_W = OUT_DEMUX_SEL_W + 2;

  // Total config bits.
  localparam int unsigned TOTAL_CFG_BITS = 1
                                         + OPCODE_WIDTH
                                         + IN_MUX_FIELD_W * MAX_FU_IN
                                         + OUT_DEMUX_FIELD_W * MAX_FU_OUT
                                         + MAX_FU_CFG_BITS;

  // Number of 32-bit config words needed.
  localparam int unsigned NUM_CFG_WORDS = (TOTAL_CFG_BITS + 31) / 32;

  // Config word index width: at least 1 bit.
  localparam int unsigned CFG_IDX_W = clog2_min1(NUM_CFG_WORDS);

  // ---------------------------------------------------------------
  // Config register: receive word-serial, pack into flat bit vector
  // ---------------------------------------------------------------
  logic [NUM_CFG_WORDS*32-1:0] cfg_flat;
  logic [CFG_IDX_W-1:0]       cfg_word_idx;

  assign cfg_ready = 1'b1;

  always_ff @(posedge clk) begin : cfg_load
    if (!rst_n) begin : cfg_reset
      cfg_flat     <= '0;
      cfg_word_idx <= '0;
    end : cfg_reset
    else begin : cfg_update
      if (cfg_valid && cfg_ready) begin : cfg_capture
        cfg_flat[cfg_word_idx*32 +: 32] <= cfg_wdata;
        // Fabric width adaptation (WA-4): config bit extraction
        // See docs/spec-rtl-width-adaptation.md
        /* verilator lint_off WIDTHTRUNC */
        if (cfg_word_idx == CFG_IDX_W'(NUM_CFG_WORDS - 1)) begin : cfg_wrap
        /* verilator lint_on WIDTHTRUNC */
          cfg_word_idx <= '0;
        end : cfg_wrap
        else begin : cfg_inc
          cfg_word_idx <= cfg_word_idx + 1;
        end : cfg_inc
      end : cfg_capture
    end : cfg_update
  end : cfg_load

  // On reset, cfg_flat is all-zero: enable=0 (PE disabled).
  // The enable=0 state gates all FU activity through the fu_slot
  // wrappers, so no explicit disconnect defaults are needed here.

  // ---------------------------------------------------------------
  // Config field extraction
  // ---------------------------------------------------------------

  // Bit cursor positions (compile-time constants).
  localparam int unsigned BIT_ENABLE    = 0;
  localparam int unsigned BIT_OPCODE    = 1;
  localparam int unsigned BIT_IN_MUX    = BIT_OPCODE + OPCODE_WIDTH;
  localparam int unsigned BIT_OUT_DEMUX = BIT_IN_MUX + IN_MUX_FIELD_W * MAX_FU_IN;
  localparam int unsigned BIT_FU_CFG    = BIT_OUT_DEMUX + OUT_DEMUX_FIELD_W * MAX_FU_OUT;

  // Enable bit.
  logic cfg_enable;
  assign cfg_enable = cfg_flat[BIT_ENABLE];
  assign pe_enable  = cfg_enable;

  // Opcode field.
  logic [OPCODE_REG_W-1:0] cfg_opcode;
  // Fabric width adaptation (WA-4): config bit extraction
  // See docs/spec-rtl-width-adaptation.md
  /* verilator lint_off WIDTHTRUNC */
  generate
    if (OPCODE_WIDTH > 0) begin : gen_opcode_extract
      assign cfg_opcode = cfg_flat[BIT_OPCODE +: OPCODE_WIDTH];
    end : gen_opcode_extract
    else begin : gen_opcode_single
      assign cfg_opcode = '0;
    end : gen_opcode_single
  endgenerate
  /* verilator lint_on WIDTHTRUNC */
  assign pe_opcode = cfg_opcode;

  // Input mux fields.
  localparam int unsigned IN_SEL_W = (IN_MUX_SEL_W > 0) ? IN_MUX_SEL_W : 1;
  logic [IN_SEL_W-1:0]     in_mux_sel        [MAX_FU_IN];
  logic [MAX_FU_IN-1:0]    in_mux_discard;
  logic [MAX_FU_IN-1:0]    in_mux_disconnect;

  // Fabric width adaptation (WA-4): config bit extraction
  // See docs/spec-rtl-width-adaptation.md
  /* verilator lint_off WIDTHTRUNC */
  generate
    genvar gi;
    for (gi = 0; gi < MAX_FU_IN; gi = gi + 1) begin : gen_in_mux_extract
      localparam int unsigned FIELD_BASE = BIT_IN_MUX + gi * IN_MUX_FIELD_W;

      if (IN_MUX_SEL_W > 0) begin : gen_in_sel
        assign in_mux_sel[gi] = cfg_flat[FIELD_BASE +: IN_MUX_SEL_W];
      end : gen_in_sel
      else begin : gen_in_sel_zero
        assign in_mux_sel[gi] = '0;
      end : gen_in_sel_zero

      assign in_mux_discard[gi]    = cfg_flat[FIELD_BASE + IN_MUX_SEL_W];
      assign in_mux_disconnect[gi] = cfg_flat[FIELD_BASE + IN_MUX_SEL_W + 1];
    end : gen_in_mux_extract
  endgenerate
  /* verilator lint_on WIDTHTRUNC */

  // Output demux fields.
  localparam int unsigned OUT_SEL_W = (OUT_DEMUX_SEL_W > 0) ? OUT_DEMUX_SEL_W : 1;
  logic [OUT_SEL_W-1:0]     out_demux_sel        [MAX_FU_OUT];
  logic [MAX_FU_OUT-1:0]    out_demux_discard;
  logic [MAX_FU_OUT-1:0]    out_demux_disconnect;

  // Fabric width adaptation (WA-4): config bit extraction
  // See docs/spec-rtl-width-adaptation.md
  /* verilator lint_off WIDTHTRUNC */
  generate
    genvar go;
    for (go = 0; go < MAX_FU_OUT; go = go + 1) begin : gen_out_demux_extract
      localparam int unsigned FIELD_BASE = BIT_OUT_DEMUX + go * OUT_DEMUX_FIELD_W;

      if (OUT_DEMUX_SEL_W > 0) begin : gen_out_sel
        assign out_demux_sel[go] = cfg_flat[FIELD_BASE +: OUT_DEMUX_SEL_W];
      end : gen_out_sel
      else begin : gen_out_sel_zero
        assign out_demux_sel[go] = '0;
      end : gen_out_sel_zero

      assign out_demux_discard[go]    = cfg_flat[FIELD_BASE + OUT_DEMUX_SEL_W];
      assign out_demux_disconnect[go] = cfg_flat[FIELD_BASE + OUT_DEMUX_SEL_W + 1];
    end : gen_out_demux_extract
  endgenerate
  /* verilator lint_on WIDTHTRUNC */

  // FU config bits.
  // Fabric width adaptation (WA-4): config bit extraction
  // See docs/spec-rtl-width-adaptation.md
  /* verilator lint_off WIDTHTRUNC */
  generate
    if (MAX_FU_CFG_BITS > 0) begin : gen_fu_cfg_extract
      assign fu_cfg_bits = cfg_flat[BIT_FU_CFG +: MAX_FU_CFG_BITS];
    end : gen_fu_cfg_extract
    else begin : gen_fu_cfg_zero
      assign fu_cfg_bits = 1'b0;
    end : gen_fu_cfg_zero
  endgenerate
  /* verilator lint_on WIDTHTRUNC */

  // ---------------------------------------------------------------
  // Input mux bank instantiation
  // ---------------------------------------------------------------
  logic [NUM_IN-1:0] mux_pe_in_ready;

  fabric_spatial_pe_mux #(
    .NUM_PE_IN  (NUM_IN),
    .NUM_FU_IN  (MAX_FU_IN),
    .DATA_WIDTH (DATA_WIDTH),
    .SEL_WIDTH  (IN_MUX_SEL_W)
  ) u_mux_bank (
    .pe_in_valid     (in_valid),
    .pe_in_data      (in_data),
    .mux_pe_in_ready (mux_pe_in_ready),
    .fu_in_valid     (fu_in_valid),
    .fu_in_ready     (fu_in_ready),
    .fu_in_data      (fu_in_data),
    .sel_cfg         (in_mux_sel),
    .discard_cfg     (in_mux_discard),
    .disconnect_cfg  (in_mux_disconnect)
  );

  // PE input ready: driven entirely by the mux bank.
  assign in_ready = mux_pe_in_ready;

  // ---------------------------------------------------------------
  // Output demux bank instantiation
  // ---------------------------------------------------------------
  logic [NUM_OUT-1:0]     demux_pe_out_valid;
  logic [DATA_WIDTH-1:0]  demux_pe_out_data [NUM_OUT];

  fabric_spatial_pe_demux #(
    .NUM_FU_OUT  (MAX_FU_OUT),
    .NUM_PE_OUT  (NUM_OUT),
    .DATA_WIDTH  (DATA_WIDTH),
    .SEL_WIDTH   (OUT_DEMUX_SEL_W)
  ) u_demux_bank (
    .fu_out_valid       (fu_out_valid),
    .fu_out_ready       (fu_out_ready),
    .fu_out_data        (fu_out_data),
    .demux_pe_out_valid (demux_pe_out_valid),
    .demux_pe_out_data  (demux_pe_out_data),
    .pe_out_ready       (out_ready),
    .sel_cfg            (out_demux_sel),
    .discard_cfg        (out_demux_discard),
    .disconnect_cfg     (out_demux_disconnect)
  );

  // PE output signals from demux bank.
  assign out_valid = demux_pe_out_valid;
  assign out_data  = demux_pe_out_data;

endmodule : fabric_spatial_pe
