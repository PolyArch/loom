// Truncation: result[OUT_WIDTH-1:0] = a[OUT_WIDTH-1:0]
module arith_trunci #(
    parameter int IN_WIDTH  = 32,
    parameter int OUT_WIDTH = 16
) (
    input  logic                a_valid,
    output logic                a_ready,
    input  logic [IN_WIDTH-1:0] a_data,
    output logic                result_valid,
    input  logic                result_ready,
    output logic [OUT_WIDTH-1:0] result_data
);
  assign result_valid = a_valid;
  assign a_ready = result_ready;

  assign result_data = a_data[OUT_WIDTH-1:0];
endmodule
