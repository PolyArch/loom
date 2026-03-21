// fu_op_store.sv -- Memory store port adapter (handshake.store).
//
// CIRCT handshake.store convention:
//   Operands (from DFG): addr (index), data (type), ctrl (none)
//   Results  (to DFG):   done (none)
//   Memory-side:         store addr + data forwarded to memory module
//
// Behavior:
//   1. Independently capture addr, data, and ctrl operands.
//   2. When all three captured and both memory channel and done output
//      are free, issue the store to memory and emit done token.

module fu_op_store #(
  parameter int unsigned ADDR_WIDTH = 32,
  parameter int unsigned DATA_WIDTH = 32
) (
  input  logic                    clk,
  input  logic                    rst_n,

  // Input 0: addr (index, ADDR_WIDTH bits)
  input  logic [ADDR_WIDTH-1:0]   in_data_0,
  input  logic                    in_valid_0,
  output logic                    in_ready_0,

  // Input 1: data (any, DATA_WIDTH bits)
  input  logic [DATA_WIDTH-1:0]   in_data_1,
  input  logic                    in_valid_1,
  output logic                    in_ready_1,

  // Input 2: ctrl (none-type trigger, width 1, data ignored)
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic                    in_data_2,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic                    in_valid_2,
  output logic                    in_ready_2,

  // Output 0: done (none-type acknowledgment, width 1)
  output logic                    out_data_0,
  output logic                    out_valid_0,
  input  logic                    out_ready_0,

  // Memory-side: forwarded addr and data for memory subsystem
  output logic [ADDR_WIDTH-1:0]   mem_addr,
  output logic [DATA_WIDTH-1:0]   mem_data,
  output logic                    mem_valid,
  input  logic                    mem_ready
);

  // -------------------------------------------------------------------
  // Input capture registers (independent capture)
  // -------------------------------------------------------------------
  logic                   addr_captured_r;
  logic [ADDR_WIDTH-1:0]  addr_val_r;
  logic                   data_captured_r;
  logic [DATA_WIDTH-1:0]  data_val_r;
  logic                   ctrl_captured_r;

  // -------------------------------------------------------------------
  // Output and memory holding registers
  // -------------------------------------------------------------------
  logic              out_valid_r;
  logic              mem_valid_r;

  assign out_valid_0 = out_valid_r;
  assign out_data_0  = 1'b0;  // none-type: data is zero

  assign mem_valid = mem_valid_r;
  assign mem_addr  = addr_val_r;
  assign mem_data  = data_val_r;

  logic out_transfer;
  assign out_transfer = out_valid_0 & out_ready_0;

  logic mem_transfer;
  assign mem_transfer = mem_valid & mem_ready;

  // -------------------------------------------------------------------
  // Input ready
  // -------------------------------------------------------------------
  assign in_ready_0 = ~addr_captured_r;
  assign in_ready_1 = ~data_captured_r;
  assign in_ready_2 = ~ctrl_captured_r;

  // -------------------------------------------------------------------
  // Main sequential logic
  // -------------------------------------------------------------------
  always_ff @(posedge clk) begin : main_seq
    if (!rst_n) begin : reset_block
      addr_captured_r <= 1'b0;
      addr_val_r      <= '0;
      data_captured_r <= 1'b0;
      data_val_r      <= '0;
      ctrl_captured_r <= 1'b0;
      out_valid_r     <= 1'b0;
      mem_valid_r     <= 1'b0;
    end : reset_block
    else begin : active_block
      // Clear outputs on transfer
      if (out_transfer)
        out_valid_r <= 1'b0;
      if (mem_transfer)
        mem_valid_r <= 1'b0;

      // Capture addr independently
      if (in_valid_0 && !addr_captured_r) begin : cap_addr
        addr_val_r      <= in_data_0;
        addr_captured_r <= 1'b1;
      end : cap_addr

      // Capture data independently
      if (in_valid_1 && !data_captured_r) begin : cap_data
        data_val_r      <= in_data_1;
        data_captured_r <= 1'b1;
      end : cap_data

      // Capture ctrl independently
      if (in_valid_2 && !ctrl_captured_r) begin : cap_ctrl
        ctrl_captured_r <= 1'b1;
      end : cap_ctrl

      // Issue store: when all three captured and outputs free
      if (addr_captured_r && data_captured_r && ctrl_captured_r &&
          !out_valid_r && !mem_valid_r) begin : issue_store
        out_valid_r     <= 1'b1;
        mem_valid_r     <= 1'b1;
        addr_captured_r <= 1'b0;
        data_captured_r <= 1'b0;
        ctrl_captured_r <= 1'b0;
      end : issue_store
    end : active_block
  end : main_seq

endmodule : fu_op_store
