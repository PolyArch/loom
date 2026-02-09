// Floating-point fused multiply-add: result = a * b + c
module math_fma #(parameter int WIDTH = 32) (
    input  logic [WIDTH-1:0] a,
    input  logic [WIDTH-1:0] b,
    input  logic [WIDTH-1:0] c,
    output logic [WIDTH-1:0] result
);
  generate
    if (WIDTH == 32) begin : g_f32
      shortreal sa, sb, sc, sr;
      always_comb begin : op
        sa = $bitstoshortreal(a);
        sb = $bitstoshortreal(b);
        sc = $bitstoshortreal(c);
        sr = sa * sb + sc;
        result = $shortrealtobits(sr);
      end
    end else if (WIDTH == 64) begin : g_f64
      real ra, rb, rc, rr;
      always_comb begin : op
        ra = $bitstoreal(a);
        rb = $bitstoreal(b);
        rc = $bitstoreal(c);
        rr = ra * rb + rc;
        result = $realtobits(rr);
      end
    end else begin : g_unsupported
      initial $fatal(1, "math_fma: unsupported WIDTH=%0d", WIDTH);
      assign result = '0;
    end
  endgenerate
endmodule
