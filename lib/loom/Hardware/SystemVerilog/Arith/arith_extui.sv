// Zero extension: result[OUT_WIDTH-1:0] = zero_extend(a[IN_WIDTH-1:0])
module arith_extui #(
    parameter int IN_WIDTH  = 16,
    parameter int OUT_WIDTH = 32
) (
    input  logic [IN_WIDTH-1:0]  a,
    output logic [OUT_WIDTH-1:0] result
);
  assign result = {{(OUT_WIDTH - IN_WIDTH){1'b0}}, a};
endmodule
