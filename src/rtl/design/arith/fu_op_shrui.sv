// fu_op_shrui.sv -- Logical (unsigned) right shift FU operation.
//
// Combinational: result = a >> b[$clog2(WIDTH)-1:0].
// The shift amount is masked to the legal range for WIDTH.
// Intrinsic latency: 0.

module fu_op_shrui #(
  parameter int unsigned WIDTH = 32
) (
  // verilator lint_off UNUSEDSIGNAL
  input  logic                clk,
  input  logic                rst_n,
  // verilator lint_on UNUSEDSIGNAL

  // Input operand A (value to shift)
  input  logic [WIDTH-1:0]    in_data_0,
  input  logic                in_valid_0,
  output logic                in_ready_0,

  // Input operand B (shift amount, only lower $clog2(WIDTH) bits used)
  // verilator lint_off UNUSEDSIGNAL
  input  logic [WIDTH-1:0]    in_data_1,
  // verilator lint_on UNUSEDSIGNAL
  input  logic                in_valid_1,
  output logic                in_ready_1,

  // Output result
  output logic [WIDTH-1:0]    out_data,
  output logic                out_valid,
  input  logic                out_ready
);

  localparam int unsigned SHAMT_WIDTH = (WIDTH > 1) ? $clog2(WIDTH) : 1;

  assign out_valid = in_valid_0 & in_valid_1;

  assign in_ready_0 = out_ready & out_valid;
  assign in_ready_1 = out_ready & out_valid;

  logic [SHAMT_WIDTH-1:0] shamt;
  assign shamt = in_data_1[SHAMT_WIDTH-1:0];

  assign out_data = in_data_0 >> shamt;

endmodule : fu_op_shrui
