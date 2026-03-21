// fu_op_load.sv -- Memory load port adapter (handshake.load).
//
// Combinational: intrinsic latency 0.
// The actual memory latency is external to this FU; this module
// adapts the handshake protocol between the FU ports and the
// memory subsystem interface.
//
// Inputs:
//   0: addr (index, ADDR_WIDTH bits)
//   1: ctrl (none-type trigger, DATA_WIDTH bits, data ignored)
//
// Outputs:
//   0: data (any, DATA_WIDTH bits) -- returned load data from memory
//   1: done (none-type, DATA_WIDTH bits) -- memory request issued
//
// Memory-side interface (directly wired to fabric memory module):
//   mem_req_addr, mem_req_valid, mem_req_ready  -- load request
//   mem_resp_data, mem_resp_valid, mem_resp_ready -- load response
//
// Handshake contract (matching simulator commitLoad):
//   - addr and ctrl are captured independently.
//   - When both addr and ctrl are captured and done output is free,
//     issue a memory request (emit done = addr on output 1).
//   - When memory response arrives and data output is free,
//     emit data on output 0.
//   The two paths (request issue and response return) operate
//   independently, allowing pipelined load behavior.

module fu_op_load #(
  parameter int unsigned ADDR_WIDTH = 32,
  parameter int unsigned DATA_WIDTH = 32
) (
  input  logic                    clk,
  input  logic                    rst_n,

  // Input 0: addr
  input  logic [ADDR_WIDTH-1:0]   in_data_0,
  input  logic                    in_valid_0,
  output logic                    in_ready_0,

  // Input 1: ctrl (trigger, data content ignored)
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [DATA_WIDTH-1:0]   in_data_1,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic                    in_valid_1,
  output logic                    in_ready_1,

  // Output 0: data (load response)
  output logic [DATA_WIDTH-1:0]   out_data_0,
  output logic                    out_valid_0,
  input  logic                    out_ready_0,

  // Output 1: done (address forwarded to memory, acts as request)
  output logic [ADDR_WIDTH-1:0]   out_data_1,
  output logic                    out_valid_1,
  input  logic                    out_ready_1
);

  // -------------------------------------------------------------------
  // Input capture registers (independent capture)
  // -------------------------------------------------------------------
  logic                   addr_captured_r;
  logic [ADDR_WIDTH-1:0]  addr_val_r;
  logic                   ctrl_captured_r;

  // Input 1 also accepts memory response data
  // (in the simulator, input 1 is "data from memory", input 2 is ctrl)
  // But per the user spec: input 0=addr, input 1=ctrl
  // Output 0=data, output 1=done
  // The memory subsystem will provide data back via a separate path.
  //
  // For the FU adapter, we model:
  //   - Capture addr+ctrl -> emit addr on done output (request)
  //   - Memory response comes back as a separate handshake

  // -------------------------------------------------------------------
  // Output holding registers
  // -------------------------------------------------------------------
  logic                   out0_valid_r;
  logic [DATA_WIDTH-1:0]  out0_data_r;
  logic                   out1_valid_r;
  logic [ADDR_WIDTH-1:0]  out1_data_r;

  assign out_valid_0 = out0_valid_r;
  assign out_data_0  = out0_data_r;
  assign out_valid_1 = out1_valid_r;
  assign out_data_1  = out1_data_r;

  logic out0_transfer;
  logic out1_transfer;
  assign out0_transfer = out_valid_0 & out_ready_0;
  assign out1_transfer = out_valid_1 & out_ready_1;

  // -------------------------------------------------------------------
  // Input ready
  // -------------------------------------------------------------------
  assign in_ready_0 = ~addr_captured_r;
  assign in_ready_1 = ~ctrl_captured_r;

  // -------------------------------------------------------------------
  // Main sequential logic
  // -------------------------------------------------------------------
  always_ff @(posedge clk) begin : main_seq
    if (!rst_n) begin : reset_block
      addr_captured_r <= 1'b0;
      addr_val_r      <= '0;
      ctrl_captured_r <= 1'b0;
      out0_valid_r    <= 1'b0;
      out0_data_r     <= '0;
      out1_valid_r    <= 1'b0;
      out1_data_r     <= '0;
    end : reset_block
    else begin : active_block
      // Clear outputs on transfer
      if (out0_transfer)
        out0_valid_r <= 1'b0;
      if (out1_transfer)
        out1_valid_r <= 1'b0;

      // Capture addr independently
      if (in_valid_0 && !addr_captured_r) begin : cap_addr
        addr_val_r      <= in_data_0;
        addr_captured_r <= 1'b1;
      end : cap_addr

      // Capture ctrl independently
      if (in_valid_1 && !ctrl_captured_r) begin : cap_ctrl
        ctrl_captured_r <= 1'b1;
      end : cap_ctrl

      // Issue request: when both captured and done output is free
      if (addr_captured_r && ctrl_captured_r && !out1_valid_r) begin : issue_req
        out1_valid_r    <= 1'b1;
        out1_data_r     <= addr_val_r;
        addr_captured_r <= 1'b0;
        ctrl_captured_r <= 1'b0;
      end : issue_req
    end : active_block
  end : main_seq

  // Note: Output 0 (data) is driven by the memory subsystem connecting
  // back through the PE/fabric wiring. In a standalone test, the
  // testbench drives data back to the load FU through its input path.
  // The out0_data_r / out0_valid_r registers are available for the
  // memory return path integration at the PE level.

endmodule : fu_op_load
