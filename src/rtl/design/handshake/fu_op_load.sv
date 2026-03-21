// fu_op_load.sv -- Memory load port adapter (handshake.load).
//
// CIRCT handshake.load convention:
//   Operands (from DFG): addr (index), ctrl (none)
//   Results  (to DFG):   data (type),  done (none)
//   Memory-side:         load address request, load data response
//
// Behavior:
//   1. Independently capture addr and ctrl operands.
//   2. When both captured and memory request channel is free,
//      issue addr on mem_addr with mem_addr_valid.
//   3. When memory responds with data on mem_data/mem_data_valid,
//      produce data on out_data_0 and done token on out_data_1.

module fu_op_load #(
  parameter int unsigned ADDR_WIDTH = 32,
  parameter int unsigned DATA_WIDTH = 32
) (
  input  logic                    clk,
  input  logic                    rst_n,

  // Input 0: addr (index, ADDR_WIDTH bits)
  input  logic [ADDR_WIDTH-1:0]   in_data_0,
  input  logic                    in_valid_0,
  output logic                    in_ready_0,

  // Input 1: ctrl (none-type trigger, width 1, data ignored)
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic                    in_data_1,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic                    in_valid_1,
  output logic                    in_ready_1,

  // Output 0: data (load response, DATA_WIDTH bits)
  output logic [DATA_WIDTH-1:0]   out_data_0,
  output logic                    out_valid_0,
  input  logic                    out_ready_0,

  // Output 1: done (none-type acknowledgment, width 1)
  output logic                    out_data_1,
  output logic                    out_valid_1,
  input  logic                    out_ready_1,

  // Memory-side: load address request
  output logic [ADDR_WIDTH-1:0]   mem_addr,
  output logic                    mem_addr_valid,
  input  logic                    mem_addr_ready,

  // Memory-side: load data response
  input  logic [DATA_WIDTH-1:0]   mem_data,
  input  logic                    mem_data_valid,
  output logic                    mem_data_ready
);

  // -------------------------------------------------------------------
  // Input capture registers (independent capture)
  // -------------------------------------------------------------------
  logic                   addr_captured_r;
  logic [ADDR_WIDTH-1:0]  addr_val_r;
  logic                   ctrl_captured_r;

  // -------------------------------------------------------------------
  // Request-side state
  // -------------------------------------------------------------------
  logic                   req_pending_r;
  logic [ADDR_WIDTH-1:0]  req_addr_r;

  // -------------------------------------------------------------------
  // Output holding registers
  // -------------------------------------------------------------------
  logic                   out0_valid_r;
  logic [DATA_WIDTH-1:0]  out0_data_r;
  logic                   out1_valid_r;

  // -------------------------------------------------------------------
  // Output port assignments
  // -------------------------------------------------------------------
  assign out_valid_0 = out0_valid_r;
  assign out_data_0  = out0_data_r;
  assign out_valid_1 = out1_valid_r;
  assign out_data_1  = 1'b0;  // none-type: data is zero

  // -------------------------------------------------------------------
  // Memory request port assignments
  // -------------------------------------------------------------------
  assign mem_addr       = req_addr_r;
  assign mem_addr_valid = req_pending_r;

  // -------------------------------------------------------------------
  // Transfer signals
  // -------------------------------------------------------------------
  logic out0_transfer;
  logic out1_transfer;
  logic mem_req_transfer;
  assign out0_transfer   = out_valid_0 & out_ready_0;
  assign out1_transfer   = out_valid_1 & out_ready_1;
  assign mem_req_transfer = mem_addr_valid & mem_addr_ready;

  // Memory response: accept when output 0 is free (or being freed)
  assign mem_data_ready = !out0_valid_r || out0_transfer;

  // -------------------------------------------------------------------
  // Input ready
  // -------------------------------------------------------------------
  assign in_ready_0 = ~addr_captured_r;
  assign in_ready_1 = ~ctrl_captured_r;

  // -------------------------------------------------------------------
  // Derived: both operands captured and request channel is free
  // -------------------------------------------------------------------
  logic can_issue;
  assign can_issue = addr_captured_r && ctrl_captured_r && !req_pending_r;

  // -------------------------------------------------------------------
  // Main sequential logic
  // -------------------------------------------------------------------
  always_ff @(posedge clk) begin : main_seq
    if (!rst_n) begin : reset_block
      addr_captured_r <= 1'b0;
      addr_val_r      <= '0;
      ctrl_captured_r <= 1'b0;
      req_pending_r   <= 1'b0;
      req_addr_r      <= '0;
      out0_valid_r    <= 1'b0;
      out0_data_r     <= '0;
      out1_valid_r    <= 1'b0;
    end : reset_block
    else begin : active_block
      // Clear outputs on transfer
      if (out0_transfer)
        out0_valid_r <= 1'b0;
      if (out1_transfer)
        out1_valid_r <= 1'b0;

      // Clear memory request on transfer
      if (mem_req_transfer)
        req_pending_r <= 1'b0;

      // Capture addr independently
      if (in_valid_0 && !addr_captured_r) begin : cap_addr
        addr_val_r      <= in_data_0;
        addr_captured_r <= 1'b1;
      end : cap_addr

      // Capture ctrl independently
      if (in_valid_1 && !ctrl_captured_r) begin : cap_ctrl
        ctrl_captured_r <= 1'b1;
      end : cap_ctrl

      // Issue memory request: when both captured and request channel free
      if (can_issue) begin : issue_req
        req_pending_r   <= 1'b1;
        req_addr_r      <= addr_val_r;
        addr_captured_r <= 1'b0;
        ctrl_captured_r <= 1'b0;
        // Emit done token (output 1) when the request is issued
        if (!out1_valid_r) begin : emit_done
          out1_valid_r <= 1'b1;
        end : emit_done
      end : issue_req

      // Accept memory response: when data comes back and output 0 is free
      if (mem_data_valid && mem_data_ready) begin : accept_resp
        out0_valid_r <= 1'b1;
        out0_data_r  <= mem_data;
      end : accept_resp
    end : active_block
  end : main_seq

endmodule : fu_op_load
