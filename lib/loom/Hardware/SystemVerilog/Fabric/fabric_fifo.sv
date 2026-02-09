//===-- fabric_fifo.sv - Parameterized FIFO module ------------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Single-input, single-output pipeline buffer with valid/ready handshake.
// Breaks combinational loops and provides backpressure.
//
// For semantics, see spec-fabric-fifo.md.
//
//===----------------------------------------------------------------------===//

module fabric_fifo #(
    parameter int DEPTH      = 2,
    parameter int DATA_WIDTH = 32,
    parameter int TAG_WIDTH  = 0,
    parameter bit BYPASSABLE = 0,
    localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH,
    // Guard against zero-width ports so DATA_WIDTH=0 compiles but $fatal fires
    localparam int SAFE_PW       = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1,
    localparam int CONFIG_WIDTH  = BYPASSABLE ? 1 : 0
) (
    input  logic                      clk,
    input  logic                      rst_n,

    // Streaming input
    input  logic                      in_valid,
    output logic                      in_ready,
    input  logic [SAFE_PW-1:0]        in_data,

    // Streaming output
    output logic                      out_valid,
    input  logic                      out_ready,
    output logic [SAFE_PW-1:0]        out_data,

    // Configuration (only meaningful when BYPASSABLE=1)
    input  logic [CONFIG_WIDTH > 0 ? CONFIG_WIDTH-1 : 0 : 0] cfg_data
);

  // -----------------------------------------------------------------------
  // Elaboration-time parameter validation (COMP_ errors)
  // -----------------------------------------------------------------------
  initial begin : param_check
    if (DEPTH == 0)
      $fatal(1, "COMP_FIFO_DEPTH_ZERO: depth must be >= 1");
    if (PAYLOAD_WIDTH <= 0)
      $fatal(1, "COMP_FIFO_INVALID_TYPE: PAYLOAD_WIDTH (DATA_WIDTH + TAG_WIDTH) must be > 0");
  end

  // -----------------------------------------------------------------------
  // Bypass mode (generate if BYPASSABLE)
  // -----------------------------------------------------------------------
  generate
    if (BYPASSABLE) begin : g_bypassable
      wire bypass_en = cfg_data[0];

      // When bypass is active, wire input directly to output
      logic                      fifo_in_valid;
      logic                      fifo_in_ready;
      logic [SAFE_PW-1:0]        fifo_in_data;
      logic                      fifo_out_valid;
      logic                      fifo_out_ready;
      logic [SAFE_PW-1:0]        fifo_out_data;

      always_comb begin : bypass_mux
        if (bypass_en) begin : active
          // Combinational pass-through
          out_valid     = in_valid;
          in_ready      = out_ready;
          out_data      = in_data;
          // FIFO is disconnected
          fifo_in_valid = 1'b0;
          fifo_in_data  = '0;
          fifo_out_ready = 1'b0;
        end else begin : normal
          // Normal FIFO path
          fifo_in_valid  = in_valid;
          in_ready       = fifo_in_ready;
          fifo_in_data   = in_data;
          out_valid      = fifo_out_valid;
          fifo_out_ready = out_ready;
          out_data       = fifo_out_data;
        end
      end

      // Internal FIFO buffer
      fabric_fifo_core #(
        .DEPTH         (DEPTH),
        .PAYLOAD_WIDTH (SAFE_PW)
      ) u_core (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (fifo_in_valid),
        .in_ready  (fifo_in_ready),
        .in_data   (fifo_in_data),
        .out_valid (fifo_out_valid),
        .out_ready (fifo_out_ready),
        .out_data  (fifo_out_data)
      );
    end else begin : g_no_bypass
      // Non-bypassable: wire directly to core
      fabric_fifo_core #(
        .DEPTH         (DEPTH),
        .PAYLOAD_WIDTH (SAFE_PW)
      ) u_core (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (in_valid),
        .in_ready  (in_ready),
        .in_data   (in_data),
        .out_valid (out_valid),
        .out_ready (out_ready),
        .out_data  (out_data)
      );
    end
  endgenerate

endmodule

// =========================================================================
// FIFO Core: circular buffer with valid/ready handshake
// =========================================================================
module fabric_fifo_core #(
    parameter int DEPTH         = 2,
    parameter int PAYLOAD_WIDTH = 32,
    localparam int SAFE_DEPTH   = (DEPTH > 0) ? DEPTH : 1,
    localparam int PTR_WIDTH    = (SAFE_DEPTH == 1) ? 1 : $clog2(SAFE_DEPTH),
    localparam int CNT_WIDTH    = $clog2(SAFE_DEPTH + 1)
) (
    input  logic                      clk,
    input  logic                      rst_n,

    input  logic                      in_valid,
    output logic                      in_ready,
    input  logic [PAYLOAD_WIDTH-1:0]  in_data,

    output logic                      out_valid,
    input  logic                      out_ready,
    output logic [PAYLOAD_WIDTH-1:0]  out_data
);

  // Storage
  logic [PAYLOAD_WIDTH-1:0] buffer [SAFE_DEPTH];

  // Pointers and count
  logic [PTR_WIDTH-1:0]  head;  // write pointer
  logic [PTR_WIDTH-1:0]  tail;  // read pointer
  logic [CNT_WIDTH-1:0]  count;

  wire full  = (count == CNT_WIDTH'(SAFE_DEPTH));
  wire empty = (count == '0);

  wire pop_pending = out_valid && out_ready;

  // Full throughput for depth>=2: accept a new write even when full
  // if a pop is happening on the same cycle (write-through).
  // Depth-1 uses strict single-slot semantics (no write-through).
  assign in_ready  = (SAFE_DEPTH > 1) ? (!full || pop_pending) : !full;
  assign out_valid = !empty;
  assign out_data  = buffer[tail];

  wire do_write = in_valid  && in_ready;
  wire do_read  = pop_pending;

  always_ff @(posedge clk or negedge rst_n) begin : seq
    if (!rst_n) begin : reset
      head  <= '0;
      tail  <= '0;
      count <= '0;
    end else begin : operate
      if (do_write) begin : write
        buffer[head] <= in_data;
        head <= (SAFE_DEPTH == 1) ? '0 :
                (head == PTR_WIDTH'(SAFE_DEPTH - 1)) ? '0 : head + 1'b1;
      end

      if (do_read) begin : read
        tail <= (SAFE_DEPTH == 1) ? '0 :
                (tail == PTR_WIDTH'(SAFE_DEPTH - 1)) ? '0 : tail + 1'b1;
      end

      case ({do_write, do_read})
        2'b10:   count <= count + 1'b1;
        2'b01:   count <= count - 1'b1;
        default: ; // 2'b00 or 2'b11: count unchanged
      endcase
    end
  end

endmodule
