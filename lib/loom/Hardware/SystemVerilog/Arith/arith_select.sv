// Ternary select: result = condition ? a : b
// condition is 1-bit (i1), a and b are WIDTH-bit
module arith_select #(parameter int WIDTH = 32) (
    input  logic             condition_valid,
    output logic             condition_ready,
    input  logic             condition_data,
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
  assign result_data      = condition_data ? a_data : b_data;
  assign result_valid     = condition_valid & a_valid & b_valid;
  assign condition_ready  = result_ready & a_valid & b_valid;
  assign a_ready          = result_ready & condition_valid & b_valid;
  assign b_ready          = result_ready & condition_valid & a_valid;
endmodule
