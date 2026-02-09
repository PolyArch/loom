// Handshake fork: broadcast one input to N outputs.
// Input consumed only when ALL outputs have accepted.
module handshake_fork #(
    parameter int NUM_OUTPUTS = 2,
    parameter int WIDTH       = 32
) (
    input  logic                              in_valid,
    output logic                              in_ready,
    input  logic [WIDTH-1:0]                  in_data,

    output logic [NUM_OUTPUTS-1:0]            out_valid,
    input  logic [NUM_OUTPUTS-1:0]            out_ready,
    output logic [NUM_OUTPUTS-1:0][WIDTH-1:0] out_data
);

  // Track which outputs have already accepted (within a handshake cycle)
  logic [NUM_OUTPUTS-1:0] done;
  logic all_done;

  assign all_done = &(done | out_ready);
  assign in_ready = all_done;

  always_comb begin : gen_outputs
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_OUTPUTS; iter_var0 = iter_var0 + 1) begin : per_out
      out_valid[iter_var0] = in_valid && !done[iter_var0];
      out_data[iter_var0]  = in_data;
    end
  end

  // NOTE: The 'done' register tracking requires clk/rst for multi-cycle
  // fork semantics. For purely combinational (eager) fork where all outputs
  // must accept simultaneously, done is always 0 and all_done = &out_ready.
  assign done = '0;

endmodule
