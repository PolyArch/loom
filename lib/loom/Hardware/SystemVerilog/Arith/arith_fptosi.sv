// Floating-point to signed integer conversion
module arith_fptosi #(
    parameter int IN_WIDTH  = 32,
    parameter int OUT_WIDTH = 32
) (
    input  logic [IN_WIDTH-1:0]  a,
    output logic [OUT_WIDTH-1:0] result
);
  generate
    if (IN_WIDTH == 32) begin : g_f32
      shortreal sa;
      always_comb begin : conv
        sa = $bitstoshortreal(a);
        result = OUT_WIDTH'($rtoi(sa));
      end
    end else if (IN_WIDTH == 64) begin : g_f64
      real ra;
      always_comb begin : conv
        ra = $bitstoreal(a);
        result = OUT_WIDTH'($rtoi(ra));
      end
    end else begin : g_unsupported
      initial $fatal(1, "arith_fptosi: unsupported IN_WIDTH=%0d", IN_WIDTH);
      assign result = '0;
    end
  endgenerate
endmodule
