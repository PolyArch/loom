// Handshake join: synchronize N inputs into one output.
// Output valid only when ALL inputs are valid.
// Data output is concatenation of all input data (MSB = input[N-1]).
module handshake_join #(
    parameter int NUM_INPUTS = 2,
    parameter int WIDTH      = 32
) (
    input  logic [NUM_INPUTS-1:0]            in_valid,
    output logic [NUM_INPUTS-1:0]            in_ready,
    input  logic [NUM_INPUTS-1:0][WIDTH-1:0] in_data,

    output logic                                          out_valid,
    input  logic                                          out_ready,
    output logic [NUM_INPUTS*WIDTH-1:0]                   out_data
);

  logic all_valid;
  assign all_valid = &in_valid;
  assign out_valid = all_valid;

  always_comb begin : gen_ready
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : per_in
      in_ready[iter_var0] = all_valid && out_ready;
    end
  end

  // Concatenate input data: in_data[N-1] at MSB, in_data[0] at LSB
  always_comb begin : concat_data
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : pack
      out_data[iter_var0*WIDTH +: WIDTH] = in_data[iter_var0];
    end
  end

endmodule
