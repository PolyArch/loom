// Arithmetic right shift (signed): result = signed(a) >>> b
module arith_shrsi #(parameter int WIDTH = 32) (
    input  logic [WIDTH-1:0] a,
    input  logic [WIDTH-1:0] b,
    output logic [WIDTH-1:0] result
);
  assign result = $signed(a) >>> b;
endmodule
