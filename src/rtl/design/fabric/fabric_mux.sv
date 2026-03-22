// fabric_mux.sv -- Configuration-time structural selector for FU bodies.
//
// Selects one of NUM_IN inputs and routes it to the output based on
// a config-time selection index.  Used inside generated FU body modules
// to implement fabric.mux operations.
//
// Config layout (low-to-high):
//   sel       : clog2(NUM_IN) bits  (when NUM_IN > 1, else 0 bits)
//   discard   : 1 bit
//   disconnect: 1 bit
//
// When disconnect=1, the output is invalid and all inputs are not ready.
// When discard=1, the selected input is consumed (ready=1) but the
// output is not driven valid.
// Otherwise, the selected input is routed to the output with full
// handshake passthrough.

module fabric_mux
  import fabric_pkg::*;
#(
  parameter int unsigned NUM_IN     = 2,
  parameter int unsigned DATA_WIDTH = 32
)(
  input  logic                    clk,
  input  logic                    rst_n,

  // --- Config port (word-serial, 1 word) ---
  input  logic                    cfg_valid,
  input  logic [31:0]             cfg_wdata,
  output logic                    cfg_ready,

  // --- Inputs (array of NUM_IN channels) ---
  input  logic [NUM_IN-1:0]       in_valid,
  output logic [NUM_IN-1:0]       in_ready,
  input  logic [DATA_WIDTH-1:0]   in_data  [0:NUM_IN-1],

  // --- Output ---
  output logic                    out_valid,
  input  logic                    out_ready,
  output logic [DATA_WIDTH-1:0]   out_data
);

  // ---------------------------------------------------------------
  // Localparams
  // ---------------------------------------------------------------
  localparam int unsigned SEL_WIDTH = clog2(NUM_IN);
  // Internal sel register width: at least 1 bit to avoid zero-width signals.
  localparam int unsigned SEL_REG_W = (SEL_WIDTH > 0) ? SEL_WIDTH : 1;

  // ---------------------------------------------------------------
  // Config register
  // ---------------------------------------------------------------
  logic [SEL_REG_W-1:0] cfg_sel;
  logic                 cfg_discard;
  logic                 cfg_disconnect;

  assign cfg_ready = 1'b1;

  always_ff @(posedge clk) begin : cfg_load
    if (!rst_n) begin : cfg_reset
      cfg_sel        <= '0;
      cfg_discard    <= 1'b0;
      cfg_disconnect <= 1'b1;  // Default: disconnected
    end : cfg_reset
    else begin : cfg_update
      if (cfg_valid && cfg_ready) begin : cfg_capture
        // Fabric width adaptation (WA-4): config bit extraction
        // See docs/spec-rtl-width-adaptation.md
        /* verilator lint_off WIDTHTRUNC */
        if (SEL_WIDTH > 0) begin : sel_unpack
          cfg_sel <= cfg_wdata[SEL_WIDTH-1:0];
        end : sel_unpack
        cfg_discard    <= cfg_wdata[SEL_WIDTH];
        cfg_disconnect <= cfg_wdata[SEL_WIDTH + 1];
        /* verilator lint_on WIDTHTRUNC */
      end : cfg_capture
    end : cfg_update
  end : cfg_load

  // ---------------------------------------------------------------
  // Combinational mux logic
  // ---------------------------------------------------------------
  logic                   sel_valid;
  logic [DATA_WIDTH-1:0]  sel_data;

  always_comb begin : mux_select
    integer iter_var0;
    sel_valid = 1'b0;
    sel_data  = '0;

    // Select the chosen input.
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : input_scan
      if (SEL_REG_W'(iter_var0) == cfg_sel) begin : match
        sel_valid = in_valid[iter_var0];
        sel_data  = in_data[iter_var0];
      end : match
    end : input_scan
  end : mux_select

  // Output handshake.
  always_comb begin : output_drive
    if (cfg_disconnect) begin : mode_disconnect
      out_valid = 1'b0;
      out_data  = '0;
    end : mode_disconnect
    else if (cfg_discard) begin : mode_discard
      // Consume input but do not produce output.
      out_valid = 1'b0;
      out_data  = '0;
    end : mode_discard
    else begin : mode_normal
      out_valid = sel_valid;
      out_data  = sel_data;
    end : mode_normal
  end : output_drive

  // Input ready generation.
  always_comb begin : ready_drive
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : ready_gen
      if (cfg_disconnect) begin : rdy_disconn
        in_ready[iter_var0] = 1'b0;
      end : rdy_disconn
      else if (SEL_REG_W'(iter_var0) == cfg_sel) begin : rdy_selected
        if (cfg_discard) begin : rdy_discard
          // Always accept from selected input in discard mode.
          in_ready[iter_var0] = 1'b1;
        end : rdy_discard
        else begin : rdy_normal
          in_ready[iter_var0] = out_ready;
        end : rdy_normal
      end : rdy_selected
      else begin : rdy_unselected
        in_ready[iter_var0] = 1'b0;
      end : rdy_unselected
    end : ready_gen
  end : ready_drive

endmodule : fabric_mux
