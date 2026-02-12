// Integer comparison with runtime-configurable predicate.
// predicate encoding (MLIR arith.cmpi):
//   0=eq, 1=ne, 2=slt, 3=sle, 4=sgt, 5=sge, 6=ult, 7=ule, 8=ugt, 9=uge
// Output is always 1 bit.
module arith_cmpi #(
    parameter int WIDTH     = 32
) (
    input  logic             a_valid,
    output logic             a_ready,
    input  logic [WIDTH-1:0] a_data,
    input  logic             b_valid,
    output logic             b_ready,
    input  logic [WIDTH-1:0] b_data,
    output logic             result_valid,
    input  logic             result_ready,
    output logic             result_data,
    input  logic [3:0]       predicate
);
  always_comb begin : cmp
    case (predicate)
      0:  result_data = (a_data == b_data);
      1:  result_data = (a_data != b_data);
      2:  result_data = ($signed(a_data) < $signed(b_data));
      3:  result_data = ($signed(a_data) <= $signed(b_data));
      4:  result_data = ($signed(a_data) > $signed(b_data));
      5:  result_data = ($signed(a_data) >= $signed(b_data));
      6:  result_data = (a_data < b_data);
      7:  result_data = (a_data <= b_data);
      8:  result_data = (a_data > b_data);
      9:  result_data = (a_data >= b_data);
      default: result_data = 1'b0;
    endcase
  end
  assign result_valid = a_valid & b_valid;
  assign a_ready      = result_ready & b_valid;
  assign b_ready      = result_ready & a_valid;
endmodule
