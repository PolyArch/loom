// Signed integer remainder: result = signed(a) % signed(b)
module arith_remsi #(parameter int WIDTH = 32) (
    input  logic             a_valid,
    output logic             a_ready,
    input  logic [WIDTH-1:0] a_data,
    input  logic             b_valid,
    output logic             b_ready,
    input  logic [WIDTH-1:0] b_data,
    output logic             result_valid,
    input  logic             result_ready,
    output logic [WIDTH-1:0] result_data
);
  assign result_data  = $signed(a_data) % $signed(b_data);
  assign result_valid = a_valid & b_valid;
  assign a_ready      = result_ready & b_valid;
  assign b_ready      = result_ready & a_valid;
endmodule
