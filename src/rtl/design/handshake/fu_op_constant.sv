// fu_op_constant.sv -- Configurable constant source (handshake.constant).
//
// Combinational: intrinsic latency 0.
//
// Input:
//   0: ctrl (none-type trigger, data ignored, WIDTH bits)
//
// Output:
//   0: value (any, WIDTH bits) -- configured constant
//
// Config:
//   cfg_value (WIDTH bits) -- the constant literal to emit
//
// Handshake contract:
//   - When ctrl is valid and output is ready, emit cfg_value.
//   - The ctrl input acts as a firing trigger; its data content is ignored.

module fu_op_constant #(
  parameter int unsigned WIDTH = 32
) (
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic                clk,
  input  logic                rst_n,
  /* verilator lint_on UNUSEDSIGNAL */

  // Input 0: ctrl (trigger, data content ignored)
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [WIDTH-1:0]    in_data_0,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic                in_valid_0,
  output logic                in_ready_0,

  // Output 0: value (constant)
  output logic [WIDTH-1:0]    out_data,
  output logic                out_valid,
  input  logic                out_ready,

  // Configuration: literal value
  input  logic [WIDTH-1:0]    cfg_value
);

  // Output valid when trigger is valid.
  assign out_valid = in_valid_0;

  // Constant value from configuration register.
  assign out_data = cfg_value;

  // Accept trigger when output can complete transfer.
  assign in_ready_0 = out_ready & out_valid;

endmodule : fu_op_constant
