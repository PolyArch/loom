// Ternary select: result = condition ? a : b
// condition is 1-bit (i1), a and b are WIDTH-bit
module arith_select #(parameter int WIDTH = 32) (
    input  logic             condition,
    input  logic [WIDTH-1:0] a,
    input  logic [WIDTH-1:0] b,
    output logic [WIDTH-1:0] result
);
  assign result = condition ? a : b;
endmodule
