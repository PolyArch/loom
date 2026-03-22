// fabric_fifo_mem.sv -- Synchronous FIFO primitive.
//
// Circular-buffer FIFO with parameterized depth and data width.
// Provides push/pop interface with full, empty, and occupancy count
// outputs.  Suitable for synthesis (inferred register or SRAM).
//
// When DEPTH == 1 the FIFO simplifies to a single-entry register.

module fabric_fifo_mem #(
  parameter int unsigned DEPTH      = 4,
  parameter int unsigned DATA_WIDTH = 32
)(
  input  logic                              clk,
  input  logic                              rst_n,

  // Write side
  input  logic                              push,
  input  logic [DATA_WIDTH-1:0]             din,

  // Read side
  input  logic                              pop,
  output logic [DATA_WIDTH-1:0]             dout,

  // Status
  output logic                              full,
  output logic                              empty,
  output logic [$clog2(DEPTH+1)-1:0]        count
);

  // ---------------------------------------------------------------
  // Localparams
  // ---------------------------------------------------------------
  localparam int unsigned CNT_W = $clog2(DEPTH + 1);
  localparam int unsigned PTR_W = (DEPTH > 1) ? $clog2(DEPTH) : 1;

  // ---------------------------------------------------------------
  // DEPTH == 1 edge case
  // ---------------------------------------------------------------
  generate
    if (DEPTH == 1) begin : gen_depth1

      logic                  occupied;
      logic [DATA_WIDTH-1:0] storage;

      always_ff @(posedge clk or negedge rst_n) begin : depth1_seq
        if (!rst_n) begin : depth1_reset
          occupied <= 1'b0;
          storage  <= '0;
        end : depth1_reset
        else begin : depth1_op
          case ({push & ~full, pop & ~empty})
            2'b10: begin : depth1_push
              storage  <= din;
              occupied <= 1'b1;
            end : depth1_push
            2'b01: begin : depth1_pop
              occupied <= 1'b0;
            end : depth1_pop
            2'b11: begin : depth1_pushpop
              storage  <= din;
              // occupied stays 1
            end : depth1_pushpop
            default: begin : depth1_noop
              // No change.
            end : depth1_noop
          endcase
        end : depth1_op
      end : depth1_seq

      assign dout  = storage;
      assign full  = occupied;
      assign empty = ~occupied;
      assign count = CNT_W'(occupied);

    end : gen_depth1

    // ---------------------------------------------------------------
    // General case: DEPTH >= 2
    // ---------------------------------------------------------------
    else begin : gen_general

      // Storage array.
      logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];

      // Read and write pointers.
      logic [PTR_W-1:0] wr_ptr;
      logic [PTR_W-1:0] rd_ptr;

      // Occupancy counter.
      logic [CNT_W-1:0] cnt;

      // Derived status.
      assign full  = (cnt == DEPTH[CNT_W-1:0]);
      assign empty = (cnt == '0);
      assign count = cnt;
      assign dout  = mem[rd_ptr];

      // Sequential logic for pointers, counter, and storage.
      always_ff @(posedge clk or negedge rst_n) begin : fifo_seq
        if (!rst_n) begin : fifo_reset
          wr_ptr <= '0;
          rd_ptr <= '0;
          cnt    <= '0;
        end : fifo_reset
        else begin : fifo_op
          case ({push & ~full, pop & ~empty})
            2'b10: begin : fifo_push_only
              mem[wr_ptr] <= din;
              wr_ptr <= (wr_ptr == PTR_W'(DEPTH - 1)) ? '0 : (wr_ptr + 1'b1);
              cnt    <= cnt + 1'b1;
            end : fifo_push_only
            2'b01: begin : fifo_pop_only
              rd_ptr <= (rd_ptr == PTR_W'(DEPTH - 1)) ? '0 : (rd_ptr + 1'b1);
              cnt    <= cnt - 1'b1;
            end : fifo_pop_only
            2'b11: begin : fifo_push_pop
              mem[wr_ptr] <= din;
              wr_ptr <= (wr_ptr == PTR_W'(DEPTH - 1)) ? '0 : (wr_ptr + 1'b1);
              rd_ptr <= (rd_ptr == PTR_W'(DEPTH - 1)) ? '0 : (rd_ptr + 1'b1);
              // cnt unchanged
            end : fifo_push_pop
            default: begin : fifo_noop
              // No change.
            end : fifo_noop
          endcase
        end : fifo_op
      end : fifo_seq

    end : gen_general
  endgenerate

endmodule : fabric_fifo_mem
