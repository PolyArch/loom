// fabric_fifo.sv -- Pipeline buffer with optional bypass.
//
// Uses fabric_fifo_mem internally for the actual storage.
// When BYPASSABLE=1, a 1-bit config register selects between
// buffered FIFO mode and combinational bypass passthrough.
// When BYPASSABLE=0, always operates in buffered mode, no config.

module fabric_fifo #(
  parameter int unsigned DEPTH      = 4,
  parameter int unsigned DATA_WIDTH = 32,
  parameter int unsigned TAG_WIDTH  = 0,
  parameter bit          BYPASSABLE = 1'b0,
  // Derived -- do not override.
  parameter int unsigned TAG_W      = (TAG_WIDTH > 0) ? TAG_WIDTH : 1
)(
  input  logic                    clk,
  input  logic                    rst_n,

  // --- Config port (only meaningful when BYPASSABLE=1) ---
  input  logic                    cfg_valid,
  input  logic [31:0]             cfg_wdata,
  output logic                    cfg_ready,

  // --- Input ---
  input  logic                    in_valid,
  output logic                    in_ready,
  input  logic [DATA_WIDTH-1:0]   in_data,
  input  logic [TAG_W-1:0]        in_tag,

  // --- Output ---
  output logic                    out_valid,
  input  logic                    out_ready,
  output logic [DATA_WIDTH-1:0]   out_data,
  output logic [TAG_W-1:0]        out_tag
);

  // ---------------------------------------------------------------
  // Combined payload width (data + tag)
  // ---------------------------------------------------------------
  localparam int unsigned PAYLOAD_WIDTH = DATA_WIDTH + TAG_W;

  // ---------------------------------------------------------------
  // Config: bypass bit (only when BYPASSABLE)
  // ---------------------------------------------------------------
  logic cfg_bypass;

  generate
    if (BYPASSABLE) begin : gen_cfg_bypass
      assign cfg_ready = 1'b1;

      always_ff @(posedge clk) begin : cfg_load
        if (!rst_n) begin : cfg_reset
          cfg_bypass <= 1'b0;
        end : cfg_reset
        else begin : cfg_update
          if (cfg_valid && cfg_ready) begin : cfg_capture
            cfg_bypass <= cfg_wdata[0];
          end : cfg_capture
        end : cfg_update
      end : cfg_load
    end : gen_cfg_bypass
    else begin : gen_no_cfg
      assign cfg_ready  = 1'b1;
      assign cfg_bypass = 1'b0;
    end : gen_no_cfg
  endgenerate

  // ---------------------------------------------------------------
  // FIFO storage instance
  // ---------------------------------------------------------------
  logic                       fifo_push;
  logic                       fifo_pop;
  logic [PAYLOAD_WIDTH-1:0]   fifo_din;
  logic [PAYLOAD_WIDTH-1:0]   fifo_dout;
  logic                       fifo_full;
  logic                       fifo_empty;

  fabric_fifo_mem #(
    .DEPTH      (DEPTH),
    .DATA_WIDTH (PAYLOAD_WIDTH)
  ) u_mem (
    .clk   (clk),
    .rst_n (rst_n),
    .push  (fifo_push),
    .din   (fifo_din),
    .pop   (fifo_pop),
    .dout  (fifo_dout),
    .full  (fifo_full),
    .empty (fifo_empty),
    .count ()
  );

  // ---------------------------------------------------------------
  // Datapath
  // ---------------------------------------------------------------
  // Pack input payload.
  logic [PAYLOAD_WIDTH-1:0] in_payload;
  assign in_payload = {in_tag, in_data};

  // Unpack output payload.
  logic [DATA_WIDTH-1:0] fifo_out_data;
  logic [TAG_W-1:0]      fifo_out_tag;
  assign fifo_out_data = fifo_dout[DATA_WIDTH-1:0];
  assign fifo_out_tag  = fifo_dout[PAYLOAD_WIDTH-1:DATA_WIDTH];

  // ---------------------------------------------------------------
  // Mux between bypass and buffered modes
  // ---------------------------------------------------------------
  generate
    if (BYPASSABLE) begin : gen_bypass_mux

      always_comb begin : bypass_logic
        if (cfg_bypass) begin : mode_bypass
          // Combinational passthrough
          out_valid  = in_valid;
          in_ready   = out_ready;
          out_data   = in_data;
          out_tag    = in_tag;
          fifo_push  = 1'b0;
          fifo_pop   = 1'b0;
          fifo_din   = '0;
        end : mode_bypass
        else begin : mode_buffered
          // Buffered FIFO mode
          fifo_din   = in_payload;
          fifo_push  = in_valid & ~fifo_full;
          fifo_pop   = ~fifo_empty & out_ready;
          in_ready   = ~fifo_full;
          out_valid  = ~fifo_empty;
          out_data   = fifo_out_data;
          out_tag    = fifo_out_tag;
        end : mode_buffered
      end : bypass_logic

    end : gen_bypass_mux
    else begin : gen_no_bypass

      // Always buffered
      assign fifo_din  = in_payload;
      assign fifo_push = in_valid & ~fifo_full;
      assign fifo_pop  = ~fifo_empty & out_ready;
      assign in_ready  = ~fifo_full;
      assign out_valid = ~fifo_empty;
      assign out_data  = fifo_out_data;
      assign out_tag   = fifo_out_tag;

    end : gen_no_bypass
  endgenerate

endmodule : fabric_fifo
