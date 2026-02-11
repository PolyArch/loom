// Zero extension: result[OUT_WIDTH-1:0] = zero_extend(a[IN_WIDTH-1:0])
module arith_extui #(
    parameter int IN_WIDTH  = 16,
    parameter int OUT_WIDTH = 32
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

  assign result_data = {{(OUT_WIDTH - IN_WIDTH){1'b0}}, a_data};
endmodule
