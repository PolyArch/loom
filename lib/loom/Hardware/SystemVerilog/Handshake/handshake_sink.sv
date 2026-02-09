// Handshake sink: always-ready consumer that discards data.
module handshake_sink #(
    parameter int WIDTH = 32
) (
    input  logic             in_valid,
    output logic             in_ready,
    input  logic [WIDTH-1:0] in_data
);

  assign in_ready = 1'b1;

endmodule
