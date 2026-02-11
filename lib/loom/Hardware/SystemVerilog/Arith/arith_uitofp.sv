// Unsigned integer to floating-point conversion
module arith_uitofp #(
    parameter int IN_WIDTH  = 32,
    parameter int OUT_WIDTH = 32
) (
    input  logic                 a_valid,
    output logic                 a_ready,
    input  logic [IN_WIDTH-1:0]  a_data,
    output logic                 result_valid,
    input  logic                 result_ready,
    output logic [OUT_WIDTH-1:0] result_data
);
  assign result_valid = a_valid;
  assign a_ready = result_ready;

  generate
    if (OUT_WIDTH == 32) begin : g_f32
      shortreal sr;
      always_comb begin : conv
        sr = shortreal'(a_data);
        result_data = $shortrealtobits(sr);
      end
    end else if (OUT_WIDTH == 64) begin : g_f64
      real rr;
      always_comb begin : conv
        rr = real'(a_data);
        result_data = $realtobits(rr);
      end
    end else begin : g_unsupported
      initial $fatal(1, "arith_uitofp: unsupported OUT_WIDTH=%0d", OUT_WIDTH);
      assign result_data = '0;
    end
  endgenerate
endmodule
