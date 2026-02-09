// Unsigned integer to floating-point conversion
module arith_uitofp #(
    parameter int IN_WIDTH  = 32,
    parameter int OUT_WIDTH = 32
) (
    input  logic [IN_WIDTH-1:0]  a,
    output logic [OUT_WIDTH-1:0] result
);
  generate
    if (OUT_WIDTH == 32) begin : g_f32
      shortreal sr;
      always_comb begin : conv
        sr = shortreal'(a);
        result = $shortrealtobits(sr);
      end
    end else if (OUT_WIDTH == 64) begin : g_f64
      real rr;
      always_comb begin : conv
        rr = real'(a);
        result = $realtobits(rr);
      end
    end else begin : g_unsupported
      initial $fatal(1, "arith_uitofp: unsupported OUT_WIDTH=%0d", OUT_WIDTH);
      assign result = '0;
    end
  endgenerate
endmodule
