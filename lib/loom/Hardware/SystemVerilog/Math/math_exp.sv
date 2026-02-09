// Floating-point exponential: result = exp(a)
module math_exp #(parameter int WIDTH = 32) (
    input  logic [WIDTH-1:0] a,
    output logic [WIDTH-1:0] result
);
  generate
    if (WIDTH == 32) begin : g_f32
      shortreal sa;
      real rr;
      always_comb begin : op
        sa = $bitstoshortreal(a);
        rr = $exp(real'(sa));
        result = $shortrealtobits(shortreal'(rr));
      end
    end else if (WIDTH == 64) begin : g_f64
      real ra, rr;
      always_comb begin : op
        ra = $bitstoreal(a);
        rr = $exp(ra);
        result = $realtobits(rr);
      end
    end else begin : g_unsupported
      initial $fatal(1, "math_exp: unsupported WIDTH=%0d", WIDTH);
      assign result = '0;
    end
  endgenerate
endmodule
