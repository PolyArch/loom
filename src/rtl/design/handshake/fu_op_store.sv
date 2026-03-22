// fu_op_store.sv -- Memory store forwarding adapter (handshake.store).
//
// Visible-graph contract matching ADGBuilder and simulator:
//   Inputs:  in_data_0 (addr, ADDR_WIDTH), in_data_1 (data, DATA_WIDTH),
//            in_data_2 (ctrl/none, 1-bit)
//   Outputs: out_data_0 = in_data_1 (forwarded data), out_data_1 = in_data_0 (forwarded addr)
//
// This is a pure forwarding FU. Store data flows through the visible fabric
// routing graph to the memory module. The FU simply synchronizes the
// operands and forwards them.
//
// Matches SimFunctionUnitCore computeOutputs for Store body type:
//   result[0] = operand[1]  (forwarded data)
//   result[1] = operand[0]  (forwarded address)

module fu_op_store #(
  parameter int unsigned ADDR_WIDTH = 32,
  parameter int unsigned DATA_WIDTH = 32
) (
  input  logic                    clk,
  input  logic                    rst_n,

  // Input 0: addr (ADDR_WIDTH)
  input  logic [ADDR_WIDTH-1:0]   in_data_0,
  input  logic                    in_valid_0,
  output logic                    in_ready_0,

  // Input 1: data (DATA_WIDTH)
  input  logic [DATA_WIDTH-1:0]   in_data_1,
  input  logic                    in_valid_1,
  output logic                    in_ready_1,

  // Input 2: ctrl (none-type trigger, width 1, data ignored)
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic                    in_data_2,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic                    in_valid_2,
  output logic                    in_ready_2,

  // Output 0: data (forwarded in_data_1, DATA_WIDTH)
  output logic [DATA_WIDTH-1:0]   out_data_0,
  output logic                    out_valid_0,
  input  logic                    out_ready_0,

  // Output 1: addr (forwarded in_data_0, ADDR_WIDTH)
  output logic [ADDR_WIDTH-1:0]   out_data_1,
  output logic                    out_valid_1,
  input  logic                    out_ready_1
);

  // All three inputs must be valid to fire.
  logic all_inputs_valid;
  assign all_inputs_valid = in_valid_0 & in_valid_1 & in_valid_2;

  // Both outputs must be accepted to complete the transfer.
  logic all_outputs_ready;
  assign all_outputs_ready = out_ready_0 & out_ready_1;

  // Fire condition: all inputs valid AND all outputs ready.
  logic fire;
  assign fire = all_inputs_valid & all_outputs_ready;

  // Combinational forwarding: result[0] = operand[1], result[1] = operand[0]
  assign out_data_0  = in_data_1;  // forwarded data
  assign out_data_1  = in_data_0;  // forwarded addr
  assign out_valid_0 = all_inputs_valid;
  assign out_valid_1 = all_inputs_valid;

  // All inputs consumed together on fire.
  assign in_ready_0 = fire;
  assign in_ready_1 = fire;
  assign in_ready_2 = fire;

endmodule : fu_op_store
