// fu_op_uitofp.sv -- Unsigned integer to floating-point conversion FU operation.
//
// Behavioral model for simulation; vendor IP under ifdef SYNTH_FP_IP.
// Supports WIDTH=32 (f32) and WIDTH=64 (f64).
// Intrinsic latency: >= 1.

module fu_op_uitofp #(
  parameter int unsigned WIDTH = 32
) (
  input  logic                clk,
  input  logic                rst_n,

  // Input operand A (unsigned integer)
  input  logic [WIDTH-1:0]    in_data_0,
  input  logic                in_valid_0,
  output logic                in_ready_0,

  // Output result (floating-point value)
  output logic [WIDTH-1:0]    out_data,
  output logic                out_valid,
  input  logic                out_ready
);

  // Handshake: accept input when valid and output can accept.
  logic fire;
  assign fire = in_valid_0 & (~out_valid | out_ready);
  assign in_ready_0 = fire;

`ifndef SYNTH_FP_IP

  // Behavioral model: interpret input as unsigned integer, convert to FP.
  logic [WIDTH-1:0] conv_result;

  generate
    if (WIDTH == 64) begin : gen_f64
      real fp_val;
      assign fp_val = $itor($unsigned(in_data_0));
      assign conv_result = $realtobits(fp_val);
    end : gen_f64
    else begin : gen_f32
      shortreal fp_val;
      assign fp_val = $itor($unsigned(in_data_0));
      assign conv_result = $shortrealtobits(fp_val);
    end : gen_f32
  endgenerate

  // Single pipeline register for latency = 1.
  always_ff @(posedge clk) begin : pipe_reg
    if (!rst_n) begin : pipe_rst
      out_valid <= 1'b0;
    end : pipe_rst
    else begin : pipe_upd
      if (fire) begin : pipe_fire
        out_data  <= conv_result;
        out_valid <= 1'b1;
      end : pipe_fire
      else if (out_ready) begin : pipe_drain
        out_valid <= 1'b0;
      end : pipe_drain
    end : pipe_upd
  end : pipe_reg

`else

  // Vendor IP instantiation placeholder.
  // TODO: Instantiate vendor unsigned-int-to-FP IP here.

  always_ff @(posedge clk) begin : ip_pipe_reg
    if (!rst_n) begin : ip_rst
      out_valid <= 1'b0;
    end : ip_rst
    else begin : ip_upd
      if (fire) begin : ip_fire
        out_data  <= {WIDTH{1'b0}};
        out_valid <= 1'b1;
      end : ip_fire
      else if (out_ready) begin : ip_drain
        out_valid <= 1'b0;
      end : ip_drain
    end : ip_upd
  end : ip_pipe_reg

`endif

endmodule : fu_op_uitofp
