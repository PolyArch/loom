// fu_op_store.sv -- Memory store port adapter (handshake.store).
//
// ADGBuilder contract (3 inputs, 2 outputs):
//   Inputs:  in_data_0 (addr, ADDR_WIDTH), in_data_1 (data, DATA_WIDTH),
//            in_data_2 (ctrl/none, 1-bit)
//   Outputs: out_data_0 (data, forwarded, DATA_WIDTH),
//            out_data_1 (addr, forwarded, ADDR_WIDTH)
//   Memory-side: store addr + wdata forwarded to memory module
//
// Behavior:
//   1. Independently capture addr, data, and ctrl operands.
//   2. When all three captured and memory channel and outputs are free,
//      issue the store to memory and forward data/addr to outputs.

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

  // Output 0: data (forwarded, DATA_WIDTH bits)
  output logic [DATA_WIDTH-1:0]   out_data_0,
  output logic                    out_valid_0,
  input  logic                    out_ready_0,

  // Output 1: addr (forwarded, ADDR_WIDTH bits)
  output logic [ADDR_WIDTH-1:0]   out_data_1,
  output logic                    out_valid_1,
  input  logic                    out_ready_1,

  // Memory-side: forwarded addr and wdata for memory subsystem
  output logic [ADDR_WIDTH-1:0]   mem_addr,
  output logic [DATA_WIDTH-1:0]   mem_wdata,
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
  logic                   out0_valid_r;
  logic [DATA_WIDTH-1:0]  out0_data_r;
  logic                   out1_valid_r;
  logic [ADDR_WIDTH-1:0]  out1_data_r;
  logic                   mem_valid_r;

  assign out_valid_0 = out0_valid_r;
  assign out_data_0  = out0_data_r;
  assign out_valid_1 = out1_valid_r;
  assign out_data_1  = out1_data_r;

  assign mem_valid = mem_valid_r;
  assign mem_addr  = addr_val_r;
  assign mem_wdata = data_val_r;

  logic out0_transfer;
  logic out1_transfer;
  assign out0_transfer = out_valid_0 & out_ready_0;
  assign out1_transfer = out_valid_1 & out_ready_1;

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
      out0_valid_r    <= 1'b0;
      out0_data_r     <= '0;
      out1_valid_r    <= 1'b0;
      out1_data_r     <= '0;
      mem_valid_r     <= 1'b0;
    end : reset_block
    else begin : active_block
      // Clear outputs on transfer
      if (out0_transfer) begin : clr_out0
        out0_valid_r <= 1'b0;
      end : clr_out0
      if (out1_transfer) begin : clr_out1
        out1_valid_r <= 1'b0;
      end : clr_out1
      if (mem_transfer) begin : clr_mem
        mem_valid_r <= 1'b0;
      end : clr_mem

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
          !out0_valid_r && !out1_valid_r && !mem_valid_r) begin : issue_store
        out0_valid_r    <= 1'b1;
        out0_data_r     <= data_val_r;
        out1_valid_r    <= 1'b1;
        out1_data_r     <= addr_val_r;
        mem_valid_r     <= 1'b1;
        addr_captured_r <= 1'b0;
        data_captured_r <= 1'b0;
        ctrl_captured_r <= 1'b0;
      end : issue_store
    end : active_block
  end : main_seq

endmodule : fu_op_store
