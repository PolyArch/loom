// Bit reversal: result[i] = a[WIDTH-1-i]
module llvm_bitreverse #(parameter int WIDTH = 32) (
    input  logic             a_valid,
    output logic             a_ready,
    input  logic [WIDTH-1:0] a_data,
    output logic             result_valid,
    input  logic             result_ready,
    output logic [WIDTH-1:0] result_data
);
  assign result_valid = a_valid;
  assign a_ready = result_ready;

  genvar gi;
  generate
    for (gi = 0; gi < WIDTH; gi++) begin : g_rev
      assign result_data[gi] = a_data[WIDTH-1-gi];
    end
  endgenerate
endmodule
