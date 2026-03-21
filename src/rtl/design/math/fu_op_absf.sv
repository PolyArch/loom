// fu_op_absf.sv -- Floating-point absolute value FU operation.
//
// Combinational: clear the sign bit of the IEEE 754 input.
// No vendor IP needed; pure bit manipulation.
// Supports WIDTH=32 (f32) and WIDTH=64 (f64).
// Intrinsic latency: 0.

module fu_op_absf #(
  parameter int unsigned WIDTH = 32
) (
  input  logic                clk,
  input  logic                rst_n,

  // Input operand A
  input  logic [WIDTH-1:0]    in_data_0,
  input  logic                in_valid_0,
  output logic                in_ready_0,

  // Output result
  output logic [WIDTH-1:0]    out_data,
  output logic                out_valid,
  input  logic                out_ready
);

  // Single-operand: output valid when input valid.
  assign out_valid = in_valid_0;

  // Backpressure: accept input only when transfer can complete.
  assign in_ready_0 = out_ready & out_valid;

  // Absolute value: clear the MSB (sign bit), leave exponent and mantissa unchanged.
  assign out_data = {1'b0, in_data_0[WIDTH-2:0]};

endmodule : fu_op_absf
