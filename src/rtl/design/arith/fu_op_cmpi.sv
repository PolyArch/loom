// fu_op_cmpi.sv -- Integer comparison FU operation.
//
// Combinational: produces a 1-bit result based on cfg_predicate.
// Predicate encoding (matches MLIR arith.cmpi):
//   0 = eq   (a == b)
//   1 = ne   (a != b)
//   2 = slt  (signed a <  signed b)
//   3 = sle  (signed a <= signed b)
//   4 = sgt  (signed a >  signed b)
//   5 = sge  (signed a >= signed b)
//   6 = ult  (unsigned a <  unsigned b)
//   7 = ule  (unsigned a <= unsigned b)
//   8 = ugt  (unsigned a >  unsigned b)
//   9 = uge  (unsigned a >= unsigned b)
//
// Intrinsic latency: 0.

module fu_op_cmpi #(
  parameter int unsigned WIDTH = 32
) (
  // verilator lint_off UNUSEDSIGNAL
  input  logic                clk,
  input  logic                rst_n,
  // verilator lint_on UNUSEDSIGNAL

  // 4-bit predicate configuration
  input  logic [3:0]          cfg_bits,

  // Input operand A
  input  logic [WIDTH-1:0]    in_data_0,
  input  logic                in_valid_0,
  output logic                in_ready_0,

  // Input operand B
  input  logic [WIDTH-1:0]    in_data_1,
  input  logic                in_valid_1,
  output logic                in_ready_1,

  // Output result (1-bit comparison result, zero-extended to WIDTH)
  output logic [WIDTH-1:0]    out_data,
  output logic                out_valid,
  input  logic                out_ready
);

  assign out_valid = in_valid_0 & in_valid_1;

  assign in_ready_0 = out_ready & out_valid;
  assign in_ready_1 = out_ready & out_valid;

  // Signed interpretations of the operands.
  logic signed [WIDTH-1:0] a_signed;
  logic signed [WIDTH-1:0] b_signed;
  assign a_signed = $signed(in_data_0);
  assign b_signed = $signed(in_data_1);

  // Comparison result (1-bit).
  logic cmp_result;

  always_comb begin : cmp_eval
    case (cfg_bits)
      4'd0:    cmp_result = (in_data_0 == in_data_1);
      4'd1:    cmp_result = (in_data_0 != in_data_1);
      4'd2:    cmp_result = (a_signed <  b_signed);
      4'd3:    cmp_result = (a_signed <= b_signed);
      4'd4:    cmp_result = (a_signed >  b_signed);
      4'd5:    cmp_result = (a_signed >= b_signed);
      4'd6:    cmp_result = (in_data_0 <  in_data_1);
      4'd7:    cmp_result = (in_data_0 <= in_data_1);
      4'd8:    cmp_result = (in_data_0 >  in_data_1);
      4'd9:    cmp_result = (in_data_0 >= in_data_1);
      default: cmp_result = 1'b0;
    endcase
  end : cmp_eval

  // Zero-extend the 1-bit result to WIDTH.
  assign out_data = {{(WIDTH-1){1'b0}}, cmp_result};

endmodule : fu_op_cmpi
