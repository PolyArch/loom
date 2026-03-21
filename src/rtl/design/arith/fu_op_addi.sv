// fu_op_addi.sv -- Integer addition FU operation.
//
// Combinational: result = a + b, masked to WIDTH.
// Intrinsic latency: 0.

module fu_op_addi #(
  parameter int unsigned WIDTH = 32
) (
  // verilator lint_off UNUSEDSIGNAL
  input  logic                clk,
  input  logic                rst_n,
  // verilator lint_on UNUSEDSIGNAL

  // Input operand A
  input  logic [WIDTH-1:0]    in_data_0,
  input  logic                in_valid_0,
  output logic                in_ready_0,

  // Input operand B
  input  logic [WIDTH-1:0]    in_data_1,
  input  logic                in_valid_1,
  output logic                in_ready_1,

  // Output result
  output logic [WIDTH-1:0]    out_data,
  output logic                out_valid,
  input  logic                out_ready
);

  // All inputs valid => output valid.
  assign out_valid = in_valid_0 & in_valid_1;

  // Backpressure: accept inputs only when transfer can complete.
  assign in_ready_0 = out_ready & out_valid;
  assign in_ready_1 = out_ready & out_valid;

  // Combinational datapath.
  assign out_data = in_data_0 + in_data_1;

endmodule : fu_op_addi
