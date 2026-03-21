// fu_op_extsi.sv -- Sign extension FU operation.
//
// Combinational: sign-extends IN_WIDTH-bit input to OUT_WIDTH-bit output.
// The MSB of the input (bit IN_WIDTH-1) is replicated into higher bits.
// Intrinsic latency: 0.

module fu_op_extsi #(
  parameter int unsigned IN_WIDTH  = 16,
  parameter int unsigned OUT_WIDTH = 32
) (
  // verilator lint_off UNUSEDSIGNAL
  input  logic                  clk,
  input  logic                  rst_n,
  // verilator lint_on UNUSEDSIGNAL

  // Input operand A (IN_WIDTH bits)
  input  logic [IN_WIDTH-1:0]   in_data_0,
  input  logic                  in_valid_0,
  output logic                  in_ready_0,

  // Output result (OUT_WIDTH bits, sign-extended)
  output logic [OUT_WIDTH-1:0]  out_data,
  output logic                  out_valid,
  input  logic                  out_ready
);

  assign out_valid  = in_valid_0;
  assign in_ready_0 = out_ready & out_valid;

  // Sign extension: replicate the MSB of the input into the upper bits.
  assign out_data = {{(OUT_WIDTH - IN_WIDTH){in_data_0[IN_WIDTH-1]}}, in_data_0};

endmodule : fu_op_extsi
