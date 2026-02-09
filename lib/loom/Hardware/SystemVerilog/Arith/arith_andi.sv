// Bitwise AND: result = a & b
module arith_andi #(parameter int WIDTH = 32) (
    input  logic [WIDTH-1:0] a,
    input  logic [WIDTH-1:0] b,
    output logic [WIDTH-1:0] result
);
  assign result = a & b;
endmodule
