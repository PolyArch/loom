// Unsigned index cast: zero-extending width conversion
module arith_index_castui #(
    parameter int IN_WIDTH  = 64,
    parameter int OUT_WIDTH = 32
) (
    input  logic [IN_WIDTH-1:0]  a,
    output logic [OUT_WIDTH-1:0] result
);
  generate
    if (OUT_WIDTH > IN_WIDTH) begin : g_extend
      assign result = {{(OUT_WIDTH - IN_WIDTH){1'b0}}, a};
    end else begin : g_truncate
      assign result = a[OUT_WIDTH-1:0];
    end
  endgenerate
endmodule
