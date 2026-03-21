// fu_op_select.sv -- 3-input mux (select) FU operation.
//
// Combinational: result = sel ? a : b
// Operand 0 = sel (i1, 1-bit condition; only bit[0] is used)
// Operand 1 = a   (true value)
// Operand 2 = b   (false value)
// Intrinsic latency: 0.

module fu_op_select #(
  parameter int unsigned WIDTH = 32
) (
  // verilator lint_off UNUSEDSIGNAL
  input  logic                clk,
  input  logic                rst_n,
  // verilator lint_on UNUSEDSIGNAL

  // Input operand 0: selector (i1, only bit[0] used)
  // verilator lint_off UNUSEDSIGNAL
  input  logic [WIDTH-1:0]    in_data_0,
  // verilator lint_on UNUSEDSIGNAL
  input  logic                in_valid_0,
  output logic                in_ready_0,

  // Input operand 1: true value (a)
  input  logic [WIDTH-1:0]    in_data_1,
  input  logic                in_valid_1,
  output logic                in_ready_1,

  // Input operand 2: false value (b)
  input  logic [WIDTH-1:0]    in_data_2,
  input  logic                in_valid_2,
  output logic                in_ready_2,

  // Output result
  output logic [WIDTH-1:0]    out_data,
  output logic                out_valid,
  input  logic                out_ready
);

  // All three inputs must be valid.
  assign out_valid = in_valid_0 & in_valid_1 & in_valid_2;

  assign in_ready_0 = out_ready & out_valid;
  assign in_ready_1 = out_ready & out_valid;
  assign in_ready_2 = out_ready & out_valid;

  // Mux: sel[0] ? a : b
  assign out_data = in_data_0[0] ? in_data_1 : in_data_2;

endmodule : fu_op_select
