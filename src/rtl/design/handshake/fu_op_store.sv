// fu_op_store.sv -- Memory store port adapter (handshake.store).
//
// Combinational: intrinsic latency 0.
// The actual memory latency is external to this FU; this module
// adapts the handshake protocol between the FU ports and the
// memory subsystem interface.
//
// Inputs:
//   0: addr (index, ADDR_WIDTH bits)
//   1: data (any, DATA_WIDTH bits)
//   2: ctrl (none-type trigger, DATA_WIDTH bits, data ignored)
//
// Output:
//   0: done (none-type, DATA_WIDTH bits) -- store request issued
//
// The store adapter captures all three inputs independently,
// then when all are captured and output is free, emits both
// addr and data to the memory subsystem (via output ports or
// direct memory-side signals wired at PE level).
//
// Matching simulator commitStore behavior:
//   - Independently capture addr, data, ctrl.
//   - When all three captured and output free, fire:
//     emit data on output 0, emit addr on output 1.
//     (The two-output model lets the PE route data/addr to memory.)
//
// For simplicity in the single-done-output model requested:
//   Output 0: done signal (data = 0, none-type acknowledgment)
//   The addr and data are forwarded to memory via separate side ports.

module fu_op_store #(
  parameter int unsigned ADDR_WIDTH = 32,
  parameter int unsigned DATA_WIDTH = 32
) (
  input  logic                    clk,
  input  logic                    rst_n,

  // Input 0: addr
  input  logic [ADDR_WIDTH-1:0]   in_data_0,
  input  logic                    in_valid_0,
  output logic                    in_ready_0,

  // Input 1: data
  input  logic [DATA_WIDTH-1:0]   in_data_1,
  input  logic                    in_valid_1,
  output logic                    in_ready_1,

  // Input 2: ctrl (trigger, data content ignored)
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [DATA_WIDTH-1:0]   in_data_2,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic                    in_valid_2,
  output logic                    in_ready_2,

  // Output 0: done (acknowledgment that store has been issued)
  output logic [DATA_WIDTH-1:0]   out_data,
  output logic                    out_valid,
  input  logic                    out_ready,

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
  // Output holding register
  // -------------------------------------------------------------------
  logic              out_valid_r;
  logic              mem_valid_r;

  assign out_valid = out_valid_r;
  assign out_data  = '0;  // none-type: data is zero

  assign mem_valid = mem_valid_r;
  assign mem_addr  = addr_val_r;
  assign mem_data  = data_val_r;

  logic out_transfer;
  assign out_transfer = out_valid & out_ready;

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
