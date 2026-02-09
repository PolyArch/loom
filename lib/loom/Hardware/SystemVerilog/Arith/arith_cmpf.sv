// Floating-point comparison with PREDICATE parameter.
// PREDICATE encoding (MLIR arith.cmpf):
//   0=false, 1=oeq, 2=ogt, 3=oge, 4=olt, 5=ole, 6=one, 7=ord,
//   8=ueq, 9=ugt, 10=uge, 11=ult, 12=ule, 13=une, 14=uno, 15=true
// Output is always 1 bit.
module arith_cmpf #(
    parameter int WIDTH     = 32,
    parameter int PREDICATE = 0
) (
    input  logic [WIDTH-1:0] a,
    input  logic [WIDTH-1:0] b,
    output logic             result
);
  generate
    if (WIDTH == 32) begin : g_f32
      shortreal sa, sb;
      logic is_nan;
      always_comb begin : cmp
        sa = $bitstoshortreal(a);
        sb = $bitstoshortreal(b);
        // NaN detection: shortreal comparison with NaN returns false
        is_nan = (sa != sa) || (sb != sb);
        case (PREDICATE)
          0:  result = 1'b0;           // false
          1:  result = !is_nan && (sa == sb); // oeq
          2:  result = !is_nan && (sa > sb);  // ogt
          3:  result = !is_nan && (sa >= sb); // oge
          4:  result = !is_nan && (sa < sb);  // olt
          5:  result = !is_nan && (sa <= sb); // ole
          6:  result = !is_nan && (sa != sb); // one
          7:  result = !is_nan;               // ord
          8:  result = is_nan || (sa == sb);  // ueq
          9:  result = is_nan || (sa > sb);   // ugt
          10: result = is_nan || (sa >= sb);  // uge
          11: result = is_nan || (sa < sb);   // ult
          12: result = is_nan || (sa <= sb);  // ule
          13: result = is_nan || (sa != sb);  // une
          14: result = is_nan;                // uno
          15: result = 1'b1;                  // true
          default: result = 1'b0;
        endcase
      end
    end else if (WIDTH == 64) begin : g_f64
      real ra, rb;
      logic is_nan;
      always_comb begin : cmp
        ra = $bitstoreal(a);
        rb = $bitstoreal(b);
        is_nan = (ra != ra) || (rb != rb);
        case (PREDICATE)
          0:  result = 1'b0;
          1:  result = !is_nan && (ra == rb);
          2:  result = !is_nan && (ra > rb);
          3:  result = !is_nan && (ra >= rb);
          4:  result = !is_nan && (ra < rb);
          5:  result = !is_nan && (ra <= rb);
          6:  result = !is_nan && (ra != rb);
          7:  result = !is_nan;
          8:  result = is_nan || (ra == rb);
          9:  result = is_nan || (ra > rb);
          10: result = is_nan || (ra >= rb);
          11: result = is_nan || (ra < rb);
          12: result = is_nan || (ra <= rb);
          13: result = is_nan || (ra != rb);
          14: result = is_nan;
          15: result = 1'b1;
          default: result = 1'b0;
        endcase
      end
    end else begin : g_unsupported
      initial $fatal(1, "arith_cmpf: unsupported WIDTH=%0d", WIDTH);
      assign result = 1'b0;
    end
  endgenerate
endmodule
