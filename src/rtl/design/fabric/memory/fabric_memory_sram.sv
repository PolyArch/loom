// fabric_memory_sram.sv -- Synthesis-friendly SRAM wrapper.
//
// Provides a simple dual-port SRAM with one read port and one write
// port.  The array is inferred by synthesis tools.  Read data appears
// one cycle after rd_en is asserted (synchronous read).
//
// When DEPTH == 0 or WIDTH == 0, the module produces no storage and
// read data is always zero (degenerate configuration guard).

module fabric_memory_sram #(
  parameter int unsigned DEPTH = 1024,
  parameter int unsigned WIDTH = 32
)(
  input  logic                              clk,
  input  logic                              rst_n,

  // --- Write port ---
  input  logic                              wr_en,
  input  logic [$clog2(DEPTH > 1 ? DEPTH : 2)-1:0] wr_addr,
  input  logic [WIDTH-1:0]                  wr_data,

  // --- Read port ---
  input  logic                              rd_en,
  input  logic [$clog2(DEPTH > 1 ? DEPTH : 2)-1:0] rd_addr,
  output logic [WIDTH-1:0]                  rd_data,
  output logic                              rd_valid
);

  // ---------------------------------------------------------------
  // Degenerate case: no storage
  // ---------------------------------------------------------------
  generate
    if (DEPTH == 0 || WIDTH == 0) begin : gen_empty

      assign rd_data  = '0;
      assign rd_valid = 1'b0;

    end : gen_empty
    else begin : gen_sram

      // Storage array -- inferred as block RAM or register file
      // depending on DEPTH and synthesis constraints.
      (* ram_style = "auto" *)
      logic [WIDTH-1:0] mem [0:DEPTH-1];

      // Synchronous read: capture address on rd_en, output one cycle later.
      logic [WIDTH-1:0] rd_data_reg;
      logic             rd_valid_reg;

      always_ff @(posedge clk or negedge rst_n) begin : sram_read
        if (!rst_n) begin : sram_read_reset
          rd_data_reg  <= '0;
          rd_valid_reg <= 1'b0;
        end : sram_read_reset
        else begin : sram_read_op
          rd_valid_reg <= rd_en;
          if (rd_en) begin : sram_read_capture
            rd_data_reg <= mem[rd_addr];
          end : sram_read_capture
        end : sram_read_op
      end : sram_read

      assign rd_data  = rd_data_reg;
      assign rd_valid = rd_valid_reg;

      // Synchronous write.
      always_ff @(posedge clk) begin : sram_write
        if (wr_en) begin : sram_write_exec
          mem[wr_addr] <= wr_data;
        end : sram_write_exec
      end : sram_write

    end : gen_sram
  endgenerate

endmodule : fabric_memory_sram
