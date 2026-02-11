// Floating-point absolute value: result = abs(a)
module math_absf #(parameter int WIDTH = 32) (
    input  logic             a_valid,
    output logic             a_ready,
    input  logic [WIDTH-1:0] a_data,
    output logic             result_valid,
    input  logic             result_ready,
    output logic [WIDTH-1:0] result_data
);
  assign result_valid = a_valid;
  assign a_ready = result_ready;

  // Clear the sign bit
  assign result_data = {1'b0, a_data[WIDTH-2:0]};
endmodule
