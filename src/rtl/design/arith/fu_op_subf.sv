// fu_op_subf.sv -- Floating-point subtraction FU operation.
//
// Behavioral model for simulation; vendor IP under ifdef SYNTH_FP_IP.
// Supports WIDTH=32 (f32) and WIDTH=64 (f64).
// Intrinsic latency: >= 1.

module fu_op_subf #(
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

  // Handshake: accept inputs when both valid and output can accept.
  logic fire;
  assign fire = in_valid_0 & in_valid_1 & (~out_valid | out_ready);
  assign in_ready_0 = fire;
  assign in_ready_1 = fire;

`ifndef SYNTH_FP_IP

  // Behavioral model: convert to real, compute, convert back.
  logic [WIDTH-1:0] result_comb;

  generate
    if (WIDTH == 64) begin : gen_f64
      real a_real, b_real, r_real;
      assign a_real = $bitstoreal(in_data_0);
      assign b_real = $bitstoreal(in_data_1);
      assign r_real = a_real - b_real;
      assign result_comb = $realtobits(r_real);
    end : gen_f64
    else if (WIDTH == 32) begin : gen_f32
      shortreal a_real, b_real, r_real;
      assign a_real = $bitstoshortreal(in_data_0);
      assign b_real = $bitstoshortreal(in_data_1);
      assign r_real = a_real - b_real;
      assign result_comb = $shortrealtobits(r_real);
    end : gen_f32
  endgenerate

  // Single pipeline register for latency = 1.
  always_ff @(posedge clk) begin : pipe_reg
    if (!rst_n) begin : pipe_rst
      out_valid <= 1'b0;
    end : pipe_rst
    else begin : pipe_upd
      if (fire) begin : pipe_fire
        out_data  <= result_comb;
        out_valid <= 1'b1;
      end : pipe_fire
      else if (out_ready) begin : pipe_drain
        out_valid <= 1'b0;
      end : pipe_drain
    end : pipe_upd
  end : pipe_reg

`else

  // Vendor IP instantiation placeholder.
  logic [WIDTH-1:0] ip_result;
  logic             ip_valid;

  // TODO: Instantiate vendor FP subtract IP here.
  assign ip_result = {WIDTH{1'b0}};
  assign ip_valid  = 1'b0;

  always_ff @(posedge clk) begin : ip_pipe_reg
    if (!rst_n) begin : ip_rst
      out_valid <= 1'b0;
    end : ip_rst
    else begin : ip_upd
      if (fire) begin : ip_fire
        out_data  <= ip_result;
        out_valid <= 1'b1;
      end : ip_fire
      else if (out_ready) begin : ip_drain
        out_valid <= 1'b0;
      end : ip_drain
    end : ip_upd
  end : ip_pipe_reg

`endif

endmodule : fu_op_subf
