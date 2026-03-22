// fu_op_trunci.sv -- Truncation FU operation.
//
// Combinational: truncates IN_WIDTH-bit input to OUT_WIDTH-bit output
// by taking the lower OUT_WIDTH bits.
// Intrinsic latency: 0.

module fu_op_trunci #(
  parameter int unsigned IN_WIDTH  = 32,
  parameter int unsigned OUT_WIDTH = 16
) (
  // verilator lint_off UNUSEDSIGNAL
  input  logic                  clk,
  input  logic                  rst_n,
  // verilator lint_on UNUSEDSIGNAL

  // Input operand A (IN_WIDTH bits, upper bits discarded by truncation)
  // verilator lint_off UNUSEDSIGNAL
  input  logic [IN_WIDTH-1:0]   in_data_0,
  // verilator lint_on UNUSEDSIGNAL
  input  logic                  in_valid_0,
  output logic                  in_ready_0,

  // Output result (OUT_WIDTH bits, truncated)
  output logic [OUT_WIDTH-1:0]  out_data,
  output logic                  out_valid,
  input  logic                  out_ready
);

  assign out_valid  = in_valid_0;
  assign in_ready_0 = out_ready & out_valid;

  // Truncation: take the lower OUT_WIDTH bits.
  assign out_data = in_data_0[OUT_WIDTH-1:0];

endmodule : fu_op_trunci
