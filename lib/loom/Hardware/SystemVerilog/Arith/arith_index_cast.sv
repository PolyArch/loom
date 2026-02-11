// Index cast: sign-extending width conversion (like extsi/trunci combined)
module arith_index_cast #(
    parameter int IN_WIDTH  = 64,
    parameter int OUT_WIDTH = 32
) (
    input  logic                 a_valid,
    output logic                 a_ready,
    input  logic [IN_WIDTH-1:0]  a_data,
    output logic                 result_valid,
    input  logic                 result_ready,
    output logic [OUT_WIDTH-1:0] result_data
);
  assign result_valid = a_valid;
  assign a_ready = result_ready;

  generate
    if (OUT_WIDTH > IN_WIDTH) begin : g_extend
      assign result_data = {{(OUT_WIDTH - IN_WIDTH){a_data[IN_WIDTH-1]}}, a_data};
    end else begin : g_truncate
      assign result_data = a_data[OUT_WIDTH-1:0];
    end
  endgenerate
endmodule
