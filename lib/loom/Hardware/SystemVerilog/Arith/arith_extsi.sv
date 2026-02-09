// Sign extension: result[OUT_WIDTH-1:0] = sign_extend(a[IN_WIDTH-1:0])
module arith_extsi #(
    parameter int IN_WIDTH  = 16,
    parameter int OUT_WIDTH = 32
) (
    input  logic [IN_WIDTH-1:0]  a,
    output logic [OUT_WIDTH-1:0] result
);
  assign result = {{(OUT_WIDTH - IN_WIDTH){a[IN_WIDTH-1]}}, a};
endmodule
