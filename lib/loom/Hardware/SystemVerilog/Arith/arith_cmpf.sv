// Floating-point comparison with PREDICATE parameter.
// PREDICATE encoding (MLIR arith.cmpf):
//   0=false, 1=oeq, 2=ogt, 3=oge, 4=olt, 5=ole, 6=one, 7=ord,
//   8=ueq, 9=ugt, 10=uge, 11=ult, 12=ule, 13=une, 14=uno, 15=true
// Output is always 1 bit.
module arith_cmpf #(
    parameter int WIDTH     = 32,
    parameter int PREDICATE = 0
) (
    input  logic             a_valid,
    output logic             a_ready,
    input  logic [WIDTH-1:0] a_data,
    input  logic             b_valid,
    output logic             b_ready,
    input  logic [WIDTH-1:0] b_data,
    output logic             result_valid,
    input  logic             result_ready,
    output logic             result_data
);
  generate
    if (WIDTH == 32) begin : g_f32
      shortreal sa, sb;
      logic is_nan;
      always_comb begin : cmp
        sa = $bitstoshortreal(a_data);
        sb = $bitstoshortreal(b_data);
        // NaN detection: shortreal comparison with NaN returns false
        is_nan = (sa != sa) || (sb != sb);
        case (PREDICATE)
          0:  result_data = 1'b0;           // false
          1:  result_data = !is_nan && (sa == sb); // oeq
          2:  result_data = !is_nan && (sa > sb);  // ogt
          3:  result_data = !is_nan && (sa >= sb); // oge
          4:  result_data = !is_nan && (sa < sb);  // olt
          5:  result_data = !is_nan && (sa <= sb); // ole
          6:  result_data = !is_nan && (sa != sb); // one
          7:  result_data = !is_nan;               // ord
          8:  result_data = is_nan || (sa == sb);  // ueq
          9:  result_data = is_nan || (sa > sb);   // ugt
          10: result_data = is_nan || (sa >= sb);  // uge
          11: result_data = is_nan || (sa < sb);   // ult
          12: result_data = is_nan || (sa <= sb);  // ule
          13: result_data = is_nan || (sa != sb);  // une
          14: result_data = is_nan;                // uno
          15: result_data = 1'b1;                  // true
          default: result_data = 1'b0;
        endcase
      end
    end else if (WIDTH == 64) begin : g_f64
      real ra, rb;
      logic is_nan;
      always_comb begin : cmp
        ra = $bitstoreal(a_data);
        rb = $bitstoreal(b_data);
        is_nan = (ra != ra) || (rb != rb);
        case (PREDICATE)
          0:  result_data = 1'b0;
          1:  result_data = !is_nan && (ra == rb);
          2:  result_data = !is_nan && (ra > rb);
          3:  result_data = !is_nan && (ra >= rb);
          4:  result_data = !is_nan && (ra < rb);
          5:  result_data = !is_nan && (ra <= rb);
          6:  result_data = !is_nan && (ra != rb);
          7:  result_data = !is_nan;
          8:  result_data = is_nan || (ra == rb);
          9:  result_data = is_nan || (ra > rb);
          10: result_data = is_nan || (ra >= rb);
          11: result_data = is_nan || (ra < rb);
          12: result_data = is_nan || (ra <= rb);
          13: result_data = is_nan || (ra != rb);
          14: result_data = is_nan;
          15: result_data = 1'b1;
          default: result_data = 1'b0;
        endcase
      end
    end else begin : g_unsupported
      initial $fatal(1, "arith_cmpf: unsupported WIDTH=%0d", WIDTH);
      assign result_data = 1'b0;
    end
  endgenerate
  assign result_valid = a_valid & b_valid;
  assign a_ready      = result_ready & b_valid;
  assign b_ready      = result_ready & a_valid;
endmodule
