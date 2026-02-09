// Floating-point absolute value: result = abs(a)
module math_absf #(parameter int WIDTH = 32) (
    input  logic [WIDTH-1:0] a,
    output logic [WIDTH-1:0] result
);
  // Clear the sign bit
  assign result = {1'b0, a[WIDTH-2:0]};
endmodule
