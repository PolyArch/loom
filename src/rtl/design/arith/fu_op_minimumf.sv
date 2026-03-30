// fu_op_minimumf.sv -- Floating-point minimum FU operation.
//
// Behavioral model for simulation; vendor IP under ifdef SYNTH_FP_IP.
// Supports WIDTH=32 (f32) and WIDTH=64 (f64).
// Intrinsic latency: 0.

module fu_op_minimumf #(
  parameter int unsigned WIDTH = 32
) (
  input  logic                clk,
  input  logic                rst_n,

  // Input operand A
  input  logic [WIDTH-1:0]    in_data_0,
  input  logic                in_valid_0,
  output logic                in_ready_0,

  // Input operand B
  input  logic [WIDTH-1:0]    in_data_1,
  input  logic                in_valid_1,
  output logic                in_ready_1,

  // Output result
  output logic [WIDTH-1:0]    out_data,
  output logic                out_valid,
  input  logic                out_ready
);

  assign out_valid = in_valid_0 & in_valid_1;
  assign in_ready_0 = out_ready & out_valid;
  assign in_ready_1 = out_ready & out_valid;

`ifndef SYNTH_FP_IP

  logic [WIDTH-1:0] result_comb;

  generate
    if (WIDTH == 64) begin : gen_f64
      real a_real, b_real;
      logic a_nan, b_nan;
      logic [WIDTH-1:0] a_bits, b_bits;
      assign a_real = $bitstoreal(in_data_0);
      assign b_real = $bitstoreal(in_data_1);
      assign a_bits = in_data_0;
      assign b_bits = in_data_1;
      assign a_nan = (a_real != a_real);
      assign b_nan = (b_real != b_real);
      assign result_comb = a_nan ? b_bits :
                           b_nan ? a_bits :
                           ((a_real <= b_real) ? a_bits : b_bits);
    end : gen_f64
    else if (WIDTH == 32) begin : gen_f32
      shortreal a_real, b_real;
      logic a_nan, b_nan;
      logic [WIDTH-1:0] a_bits, b_bits;
      assign a_real = $bitstoshortreal(in_data_0);
      assign b_real = $bitstoshortreal(in_data_1);
      assign a_bits = in_data_0;
      assign b_bits = in_data_1;
      assign a_nan = (a_real != a_real);
      assign b_nan = (b_real != b_real);
      assign result_comb = a_nan ? b_bits :
                           b_nan ? a_bits :
                           ((a_real <= b_real) ? a_bits : b_bits);
    end : gen_f32
  endgenerate

  always_comb begin : comb_out
    out_data = result_comb;
  end : comb_out

`else

  logic [WIDTH-1:0] ip_result;
  logic             ip_valid;

  // TODO: Instantiate vendor FP minimum IP here.
  assign ip_result = {WIDTH{1'b0}};
  assign ip_valid  = 1'b0;
  assign out_data  = ip_result;
  assign out_valid = in_valid_0 & in_valid_1 & ip_valid;

`endif

endmodule : fu_op_minimumf
