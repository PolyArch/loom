// Logical right shift (unsigned): result = a >> b
module arith_shrui #(parameter int WIDTH = 32) (
    input  logic [WIDTH-1:0] a,
    input  logic [WIDTH-1:0] b,
    output logic [WIDTH-1:0] result
);
  assign result = a >> b;
endmodule
