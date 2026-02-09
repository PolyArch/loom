// Handshake constant: emit a configured constant value on each control token.
module handshake_constant #(
    parameter int WIDTH = 32
) (
    // Control input (trigger)
    input  logic             ctrl_valid,
    output logic             ctrl_ready,

    // Constant output
    output logic             out_valid,
    input  logic             out_ready,
    output logic [WIDTH-1:0] out_data,

    // Configuration: the constant value
    input  logic [WIDTH-1:0] cfg_value
);

  assign out_valid  = ctrl_valid;
  assign out_data   = cfg_value;
  assign ctrl_ready = out_ready;

endmodule
