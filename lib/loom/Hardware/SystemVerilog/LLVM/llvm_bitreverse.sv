// Bit reversal: result[i] = a[WIDTH-1-i]
module llvm_bitreverse #(parameter int WIDTH = 32) (
    input  logic [WIDTH-1:0] a,
    output logic [WIDTH-1:0] result
);
  genvar gi;
  generate
    for (gi = 0; gi < WIDTH; gi++) begin : g_rev
      assign result[gi] = a[WIDTH-1-gi];
    end
  endgenerate
endmodule
