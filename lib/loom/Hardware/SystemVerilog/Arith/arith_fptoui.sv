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
      longint li;
      always_comb begin : conv
        sa = $bitstoshortreal(a);
        li = longint'(sa);
        result = (li < 0) ? '0 : OUT_WIDTH'(li);
      end
    end else if (IN_WIDTH == 64) begin : g_f64
      real ra;
      longint li;
      always_comb begin : conv
        ra = $bitstoreal(a);
        li = longint'(ra);
        result = (li < 0) ? '0 : OUT_WIDTH'(li);
      end
    end else begin : g_unsupported
      initial $fatal(1, "arith_fptoui: unsupported IN_WIDTH=%0d", IN_WIDTH);
      assign result = '0;
    end
  endgenerate
endmodule
