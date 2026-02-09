// Handshake conditional branch: route data to true or false output
// based on a condition signal.
// Both condition and data must be valid to fire (partial produce).
module handshake_cond_br #(
    parameter int WIDTH = 32
) (
    // Condition input
    input  logic             cond_valid,
    output logic             cond_ready,
    input  logic             cond_data,

    // Data input
    input  logic             in_valid,
    output logic             in_ready,
    input  logic [WIDTH-1:0] in_data,

    // True output (when cond_data == 1)
    output logic             true_valid,
    input  logic             true_ready,
    output logic [WIDTH-1:0] true_data,

    // False output (when cond_data == 0)
    output logic             false_valid,
    input  logic             false_ready,
    output logic [WIDTH-1:0] false_data
);

  logic both_valid;
  assign both_valid = cond_valid && in_valid;

  // Route to true or false output
  assign true_valid  = both_valid && cond_data;
  assign false_valid = both_valid && !cond_data;

  assign true_data  = in_data;
  assign false_data = in_data;

  // Consume inputs when the selected output accepts
  logic selected_ready;
  assign selected_ready = cond_data ? true_ready : false_ready;

  assign cond_ready = both_valid && selected_ready;
  assign in_ready   = both_valid && selected_ready;

endmodule
