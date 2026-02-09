// Handshake mux: select one of N data inputs based on a select signal.
// The select input and chosen data input are consumed together.
// Non-selected data inputs are not consumed (partial consume).
module handshake_mux #(
    parameter int NUM_INPUTS = 2,
    parameter int WIDTH      = 32,
    localparam int SEL_WIDTH = (NUM_INPUTS > 1) ? $clog2(NUM_INPUTS) : 1
) (
    // Select input
    input  logic                              sel_valid,
    output logic                              sel_ready,
    input  logic [SEL_WIDTH-1:0]              sel_data,

    // Data inputs
    input  logic [NUM_INPUTS-1:0]            in_valid,
    output logic [NUM_INPUTS-1:0]            in_ready,
    input  logic [NUM_INPUTS-1:0][WIDTH-1:0] in_data,

    // Output
    output logic                              out_valid,
    input  logic                              out_ready,
    output logic [WIDTH-1:0]                  out_data
);

  logic selected_valid;
  assign selected_valid = (sel_data < SEL_WIDTH'(NUM_INPUTS)) ?
                          in_valid[sel_data] : 1'b0;

  assign out_valid = sel_valid && selected_valid;
  assign out_data  = in_data[sel_data];

  assign sel_ready = out_valid && out_ready;

  always_comb begin : gen_ready
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : per_in
      in_ready[iter_var0] = out_valid && out_ready &&
                            (sel_data == SEL_WIDTH'(iter_var0));
    end
  end

endmodule
