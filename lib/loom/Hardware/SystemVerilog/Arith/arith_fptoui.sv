// Floating-point to unsigned integer conversion (clamp negative to 0)
module arith_fptoui #(
    parameter int IN_WIDTH  = 32,
    parameter int OUT_WIDTH = 32
) (
    input  logic [IN_WIDTH-1:0]  a,
    output logic [OUT_WIDTH-1:0] result
);
  generate
    if (IN_WIDTH == 32) begin : g_f32
      shortreal sa;
      int signed ival;
      always_comb begin : conv
        sa = $bitstoshortreal(a);
        ival = $rtoi(sa);
        result = (ival < 0) ? '0 : OUT_WIDTH'(unsigned'(ival));
      end
    end else if (IN_WIDTH == 64) begin : g_f64
      real ra;
      int signed ival;
      always_comb begin : conv
        ra = $bitstoreal(a);
        ival = $rtoi(ra);
        result = (ival < 0) ? '0 : OUT_WIDTH'(unsigned'(ival));
      end
    end else begin : g_unsupported
      initial $fatal(1, "arith_fptoui: unsupported IN_WIDTH=%0d", IN_WIDTH);
      assign result = '0;
    end
  endgenerate
endmodule
