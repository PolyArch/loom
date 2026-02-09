// Truncation: result[OUT_WIDTH-1:0] = a[OUT_WIDTH-1:0]
module arith_trunci #(
    parameter int IN_WIDTH  = 32,
    parameter int OUT_WIDTH = 16
) (
    input  logic [IN_WIDTH-1:0]  a,
    output logic [OUT_WIDTH-1:0] result
);
  assign result = a[OUT_WIDTH-1:0];
endmodule
