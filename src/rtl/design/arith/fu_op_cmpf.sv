// fu_op_cmpf.sv -- Floating-point comparison FU operation.
//
// Combinational: produces a 1-bit result based on cfg_bits predicate.
// Behavioral model for simulation; vendor IP under ifdef SYNTH_FP_IP.
// Supports WIDTH=32 (f32) and WIDTH=64 (f64).
//
// Predicate encoding (matches MLIR arith.cmpf):
//   0  = false  (always false)
//   1  = oeq    (ordered and equal)
//   2  = ogt    (ordered and greater than)
//   3  = oge    (ordered and greater than or equal)
//   4  = olt    (ordered and less than)
//   5  = ole    (ordered and less than or equal)
//   6  = one    (ordered and not equal)
//   7  = ord    (ordered, no NaNs)
//   8  = ueq    (unordered or equal)
//   9  = ugt    (unordered or greater than)
//   10 = uge    (unordered or greater than or equal)
//   11 = ult    (unordered or less than)
//   12 = ule    (unordered or less than or equal)
//   13 = une    (unordered or not equal)
//   14 = uno    (unordered, either is NaN)
//   15 = true   (always true)
//
// Intrinsic latency: 0.

module fu_op_cmpf #(
  parameter int unsigned WIDTH = 32
) (
  input  logic                clk,
  input  logic                rst_n,

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

`ifndef SYNTH_FP_IP

  // Behavioral model: NaN detection and ordered/unordered comparison.

  // IEEE 754 field extraction for NaN detection.
  localparam int unsigned EXP_BITS = (WIDTH == 64) ? 11 : 8;
  localparam int unsigned MAN_BITS = (WIDTH == 64) ? 52 : 23;

  logic a_is_nan, b_is_nan, either_nan;
  logic a_exp_all_ones, b_exp_all_ones;
  logic a_man_nonzero, b_man_nonzero;

  assign a_exp_all_ones = &in_data_0[WIDTH-2 -: EXP_BITS];
  assign b_exp_all_ones = &in_data_1[WIDTH-2 -: EXP_BITS];
  assign a_man_nonzero  = |in_data_0[MAN_BITS-1:0];
  assign b_man_nonzero  = |in_data_1[MAN_BITS-1:0];
  assign a_is_nan       = a_exp_all_ones & a_man_nonzero;
  assign b_is_nan       = b_exp_all_ones & b_man_nonzero;
  assign either_nan     = a_is_nan | b_is_nan;

  // Real-type comparison for ordered predicates.
  logic cmp_eq, cmp_gt, cmp_lt;

  generate
    if (WIDTH == 64) begin : gen_f64_cmp
      real a_real, b_real;
      assign a_real = $bitstoreal(in_data_0);
      assign b_real = $bitstoreal(in_data_1);
      assign cmp_eq = (a_real == b_real);
      assign cmp_gt = (a_real >  b_real);
      assign cmp_lt = (a_real <  b_real);
    end : gen_f64_cmp
    else if (WIDTH == 32) begin : gen_f32_cmp
      shortreal a_real, b_real;
      assign a_real = $bitstoshortreal(in_data_0);
      assign b_real = $bitstoshortreal(in_data_1);
      assign cmp_eq = (a_real == b_real);
      assign cmp_gt = (a_real >  b_real);
      assign cmp_lt = (a_real <  b_real);
    end : gen_f32_cmp
  endgenerate

  // Derived ordered predicates.
  logic ord_eq, ord_gt, ord_ge, ord_lt, ord_le, ord_ne, is_ord;
  assign is_ord = ~either_nan;
  assign ord_eq = is_ord & cmp_eq;
  assign ord_gt = is_ord & cmp_gt;
  assign ord_ge = is_ord & (cmp_gt | cmp_eq);
  assign ord_lt = is_ord & cmp_lt;
  assign ord_le = is_ord & (cmp_lt | cmp_eq);
  assign ord_ne = is_ord & ~cmp_eq;

  // Comparison result (1-bit).
  logic cmp_result;

  always_comb begin : cmp_eval
    case (cfg_bits)
      4'd0:    cmp_result = 1'b0;                       // false
      4'd1:    cmp_result = ord_eq;                      // oeq
      4'd2:    cmp_result = ord_gt;                      // ogt
      4'd3:    cmp_result = ord_ge;                      // oge
      4'd4:    cmp_result = ord_lt;                      // olt
      4'd5:    cmp_result = ord_le;                      // ole
      4'd6:    cmp_result = ord_ne;                      // one
      4'd7:    cmp_result = is_ord;                      // ord
      4'd8:    cmp_result = either_nan | cmp_eq;         // ueq
      4'd9:    cmp_result = either_nan | cmp_gt;         // ugt
      4'd10:   cmp_result = either_nan | cmp_gt | cmp_eq; // uge
      4'd11:   cmp_result = either_nan | cmp_lt;         // ult
      4'd12:   cmp_result = either_nan | cmp_lt | cmp_eq; // ule
      4'd13:   cmp_result = either_nan | ~cmp_eq;        // une
      4'd14:   cmp_result = either_nan;                  // uno
      4'd15:   cmp_result = 1'b1;                        // true
      default: cmp_result = 1'b0;
    endcase
  end : cmp_eval

`else

  // Vendor IP instantiation placeholder for FP compare.
  logic cmp_result;

  // TODO: Instantiate vendor FP compare IP here.
  assign cmp_result = 1'b0;

`endif

  // Zero-extend the 1-bit result to WIDTH.
  assign out_data = {{(WIDTH-1){1'b0}}, cmp_result};

endmodule : fu_op_cmpf
