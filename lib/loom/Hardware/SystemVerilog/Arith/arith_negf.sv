// Floating-point negation: result = -a
// Uses shortreal for 32-bit, real for 64-bit. Not synthesizable.
module arith_negf #(parameter int WIDTH = 32) (
    input  logic [WIDTH-1:0] a,
    output logic [WIDTH-1:0] result
);
  generate
    if (WIDTH == 32) begin : g_f32
      shortreal sa, sr;
      always_comb begin : neg
        sa = $bitstoshortreal(a);
        sr = -sa;
        result = $shortrealtobits(sr);
      end
    end else if (WIDTH == 64) begin : g_f64
      real ra, rr;
      always_comb begin : neg
        ra = $bitstoreal(a);
        rr = -ra;
        result = $realtobits(rr);
      end
    end else begin : g_unsupported
      initial $fatal(1, "arith_negf: unsupported WIDTH=%0d (must be 32 or 64)", WIDTH);
      assign result = '0;
    end
  endgenerate
endmodule
