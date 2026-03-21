// fabric_temporal_pe_regfile.sv -- Register file for temporal PE.
//
// Implements NUM_REG registers, each REG_FIFO_DEPTH deep as a FIFO.
// Supports:
//   - Read: by operand config reg_idx (FIFO front, non-destructive peek).
//           Actual pop occurs on FU fire via rd_consume.
//   - Write: by result config is_reg/reg_idx (FIFO push).
//
// When NUM_REG == 0, this module is structurally inert (all outputs tied off).

module fabric_temporal_pe_regfile
  import fabric_pkg::*;
#(
  parameter int unsigned NUM_REG        = 4,
  parameter int unsigned REG_FIFO_DEPTH = 4,
  parameter int unsigned DATA_WIDTH     = 32,
  parameter int unsigned TAG_WIDTH      = 4,
  // Number of read ports (one per MAX_FU_IN operand that may source from reg)
  parameter int unsigned NUM_RD_PORTS   = 2,
  // Number of write ports (one per MAX_FU_OUT result that may target a reg)
  parameter int unsigned NUM_WR_PORTS   = 2
)(
  input  logic        clk,
  input  logic        rst_n,

  // --- Read ports (non-destructive peek at FIFO front) ---
  input  logic [REG_IDX_W-1:0]   rd_reg_idx   [NUM_RD_PORTS],
  input  logic [NUM_RD_PORTS-1:0] rd_enable,
  output logic [DATA_WIDTH-1:0]  rd_data      [NUM_RD_PORTS],
  output logic [NUM_RD_PORTS-1:0] rd_valid,    // FIFO not empty for requested reg

  // --- Read consume (pop FIFO front on FU fire) ---
  // rd_consume_en: per-register pop strobe (indexed by register number)
  input  logic [EFF_NUM_REG-1:0]  rd_consume_en,

  // --- Write ports (push to FIFO tail) ---
  input  logic [REG_IDX_W-1:0]   wr_reg_idx   [NUM_WR_PORTS],
  input  logic [NUM_WR_PORTS-1:0] wr_enable,
  input  logic [DATA_WIDTH-1:0]  wr_data      [NUM_WR_PORTS],
  output logic [NUM_WR_PORTS-1:0] wr_ready     // FIFO not full for requested reg
);

  // ---------------------------------------------------------------
  // Derived widths
  // ---------------------------------------------------------------
  localparam int unsigned REG_IDX_W   = clog2(NUM_REG);
  localparam int unsigned EFF_NUM_REG = (NUM_REG > 0) ? NUM_REG : 1;
  localparam int unsigned EFF_REG_IDX = (REG_IDX_W > 0) ? REG_IDX_W : 1;

  // ---------------------------------------------------------------
  // Generate: NUM_REG == 0 => tie off all ports
  // ---------------------------------------------------------------
  generate
    if (NUM_REG == 0) begin : gen_no_regs

      always_comb begin : tie_off_rd
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < NUM_RD_PORTS; iter_var0 = iter_var0 + 1) begin : tie_rd
          rd_data[iter_var0]  = '0;
        end : tie_rd
      end : tie_off_rd
      assign rd_valid = '0;
      assign wr_ready = '0;

    end : gen_no_regs
    else begin : gen_regs

      // FIFO instances: one per register
      logic                  fifo_push   [0:NUM_REG-1];
      logic [DATA_WIDTH-1:0] fifo_din    [0:NUM_REG-1];
      logic                  fifo_pop    [0:NUM_REG-1];
      logic [DATA_WIDTH-1:0] fifo_dout   [0:NUM_REG-1];
      logic                  fifo_full   [0:NUM_REG-1];
      logic                  fifo_empty  [0:NUM_REG-1];

      genvar g_reg;
      for (g_reg = 0; g_reg < NUM_REG; g_reg = g_reg + 1) begin : gen_fifo
        fabric_fifo_mem #(
          .DEPTH      (REG_FIFO_DEPTH),
          .DATA_WIDTH (DATA_WIDTH)
        ) u_reg_fifo (
          .clk   (clk),
          .rst_n (rst_n),
          .push  (fifo_push[g_reg]),
          .din   (fifo_din[g_reg]),
          .pop   (fifo_pop[g_reg]),
          .dout  (fifo_dout[g_reg]),
          .full  (fifo_full[g_reg]),
          .empty (fifo_empty[g_reg]),
          .count ()
        );
      end : gen_fifo

      // ---------------------------------------------------------------
      // Read port logic: peek at FIFO front
      // ---------------------------------------------------------------
      always_comb begin : read_logic
        integer iter_var0;
        integer iter_var1;

        for (iter_var0 = 0; iter_var0 < NUM_RD_PORTS; iter_var0 = iter_var0 + 1) begin : per_rd
          rd_data[iter_var0]  = '0;
          rd_valid[iter_var0] = 1'b0;

          if (rd_enable[iter_var0]) begin : rd_active
            for (iter_var1 = 0; iter_var1 < NUM_REG; iter_var1 = iter_var1 + 1) begin : scan_reg
              if (EFF_REG_IDX'(iter_var1) == rd_reg_idx[iter_var0]) begin : reg_match
                rd_data[iter_var0]  = fifo_dout[iter_var1];
                rd_valid[iter_var0] = ~fifo_empty[iter_var1];
              end : reg_match
            end : scan_reg
          end : rd_active
        end : per_rd
      end : read_logic

      // ---------------------------------------------------------------
      // Read consume: pop from FIFO
      // ---------------------------------------------------------------
      always_comb begin : consume_logic
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < NUM_REG; iter_var0 = iter_var0 + 1) begin : per_reg_pop
          fifo_pop[iter_var0] = rd_consume_en[iter_var0] & ~fifo_empty[iter_var0];
        end : per_reg_pop
      end : consume_logic

      // ---------------------------------------------------------------
      // Write port logic: push to FIFO
      //
      // If multiple write ports target the same register in the same
      // cycle, only the lowest-indexed write port wins.
      // ---------------------------------------------------------------
      always_comb begin : write_logic
        integer iter_var0;
        integer iter_var1;

        for (iter_var0 = 0; iter_var0 < NUM_REG; iter_var0 = iter_var0 + 1) begin : per_reg_wr
          fifo_push[iter_var0] = 1'b0;
          fifo_din[iter_var0]  = '0;
        end : per_reg_wr

        for (iter_var0 = 0; iter_var0 < NUM_WR_PORTS; iter_var0 = iter_var0 + 1) begin : per_wr
          wr_ready[iter_var0] = 1'b0;
          if (wr_enable[iter_var0]) begin : wr_active
            for (iter_var1 = 0; iter_var1 < NUM_REG; iter_var1 = iter_var1 + 1) begin : scan_wr_reg
              if (EFF_REG_IDX'(iter_var1) == wr_reg_idx[iter_var0]) begin : wr_match
                wr_ready[iter_var0] = ~fifo_full[iter_var1];
                if (!fifo_push[iter_var1]) begin : first_writer
                  fifo_push[iter_var1] = ~fifo_full[iter_var1];
                  fifo_din[iter_var1]  = wr_data[iter_var0];
                end : first_writer
              end : wr_match
            end : scan_wr_reg
          end : wr_active
        end : per_wr
      end : write_logic

    end : gen_regs
  endgenerate

endmodule : fabric_temporal_pe_regfile
