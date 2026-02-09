// Floating-point base-2 logarithm: result = log2(a)
// Uses log10(a)/log10(2.0) since SV does not provide $log2 for reals.
module math_log2 #(parameter int WIDTH = 32) (
    input  logic [WIDTH-1:0] a,
    output logic [WIDTH-1:0] result
);
  generate
    if (WIDTH == 32) begin : g_f32
      shortreal sa;
      real rr;
      always_comb begin : op
        sa = $bitstoshortreal(a);
        rr = $ln(real'(sa)) / $ln(2.0);
        result = $shortrealtobits(shortreal'(rr));
      end
    end else if (WIDTH == 64) begin : g_f64
      real ra, rr;
      always_comb begin : op
        ra = $bitstoreal(a);
        rr = $ln(ra) / $ln(2.0);
        result = $realtobits(rr);
      end
    end else begin : g_unsupported
      initial $fatal(1, "math_log2: unsupported WIDTH=%0d", WIDTH);
      assign result = '0;
    end
  endgenerate
endmodule
