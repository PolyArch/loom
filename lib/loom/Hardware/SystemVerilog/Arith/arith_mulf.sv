// Floating-point multiplication: result = a * b
module arith_mulf #(parameter int WIDTH = 32) (
    input  logic [WIDTH-1:0] a,
    input  logic [WIDTH-1:0] b,
    output logic [WIDTH-1:0] result
);
  generate
    if (WIDTH == 32) begin : g_f32
      shortreal sa, sb, sr;
      always_comb begin : mul
        sa = $bitstoshortreal(a);
        sb = $bitstoshortreal(b);
        sr = sa * sb;
        result = $shortrealtobits(sr);
      end
    end else if (WIDTH == 64) begin : g_f64
      real ra, rb, rr;
      always_comb begin : mul
        ra = $bitstoreal(a);
        rb = $bitstoreal(b);
        rr = ra * rb;
        result = $realtobits(rr);
      end
    end else begin : g_unsupported
      initial $fatal(1, "arith_mulf: unsupported WIDTH=%0d", WIDTH);
      assign result = '0;
    end
  endgenerate
endmodule
