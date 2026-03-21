// fu_op_log2.sv -- Transcendental FP log-base-2 (Tier 3 placeholder).
//
// This module has no portable synthesizable implementation.
// It requires --fp-ip-profile to provide a vendor IP library.
// If this file is included in a build without a vendor profile,
// elaboration will fail with an error.

module fu_op_log2 #(
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

  // Placeholder: no implementation available without vendor IP.
  initial begin : placeholder_error
    $error("Transcendental FP op requires --fp-ip-profile");
  end : placeholder_error

  assign in_ready_0 = 1'b0;
  assign out_data   = {WIDTH{1'b0}};
  assign out_valid  = 1'b0;

endmodule : fu_op_log2
