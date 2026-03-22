// fu_op_bitreverse.sv -- Bit reversal FU operation.
//
// Combinational: reverses the bit order of the input.
// result[i] = in[WIDTH-1-i] for all i in [0, WIDTH-1].
// Pure wiring, no logic gates needed.
// Intrinsic latency: 0.

module fu_op_bitreverse #(
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

  // Output result (bit-reversed)
  output logic [WIDTH-1:0]    out_data,
  output logic                out_valid,
  input  logic                out_ready
);

  assign out_valid  = in_valid_0;
  assign in_ready_0 = out_ready & out_valid;

  // Bit reversal via generate loop (pure wiring).
  genvar gen_i;
  generate
    for (gen_i = 0; gen_i < WIDTH; gen_i = gen_i + 1) begin : gen_reverse
      assign out_data[gen_i] = in_data_0[WIDTH - 1 - gen_i];
    end : gen_reverse
  endgenerate

endmodule : fu_op_bitreverse
