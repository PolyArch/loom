// Integer comparison with PREDICATE parameter.
// PREDICATE encoding (MLIR arith.cmpi):
//   0=eq, 1=ne, 2=slt, 3=sle, 4=sgt, 5=sge, 6=ult, 7=ule, 8=ugt, 9=uge
// Output is always 1 bit.
module arith_cmpi #(
    parameter int WIDTH     = 32,
    parameter int PREDICATE = 0
) (
    input  logic [WIDTH-1:0] a,
    input  logic [WIDTH-1:0] b,
    output logic             result
);
  always_comb begin : cmp
    case (PREDICATE)
      0:  result = (a == b);
      1:  result = (a != b);
      2:  result = ($signed(a) < $signed(b));
      3:  result = ($signed(a) <= $signed(b));
      4:  result = ($signed(a) > $signed(b));
      5:  result = ($signed(a) >= $signed(b));
      6:  result = (a < b);
      7:  result = (a <= b);
      8:  result = (a > b);
      9:  result = (a >= b);
      default: result = 1'b0;
    endcase
  end
endmodule
