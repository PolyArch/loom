// fu_op_join.sv -- Configurable-mask synchronizer (handshake.join).
//
// Combinational: intrinsic latency 0.
//
// Inputs:
//   0..NUM_IN-1: input ports (none type, data ignored)
//
// Output:
//   0: out (none type, fires when all mask-selected inputs are valid)
//
// Config:
//   cfg_join_mask (NUM_IN bits) -- bit i=1 means input i participates
//
// Handshake contract:
//   - Output is valid when all mask-selected inputs are valid.
//   - Only mask-selected inputs are consumed (ready asserted) on transfer.
//   - Non-selected inputs are always shown ready (consumed/ignored).

module fu_op_join #(
  parameter int unsigned NUM_IN = 2,
  parameter int unsigned WIDTH  = 32
) (
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic                    clk,
  input  logic                    rst_n,
  /* verilator lint_on UNUSEDSIGNAL */

  // Input data ports (none-type, data content is ignored)
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [NUM_IN-1:0][WIDTH-1:0] in_data,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic [NUM_IN-1:0]            in_valid,
  output logic [NUM_IN-1:0]            in_ready,

  // Output port (none type, data is zero)
  output logic [WIDTH-1:0]        out_data,
  output logic                    out_valid,
  input  logic                    out_ready,

  // Configuration: join mask
  input  logic [NUM_IN-1:0]       cfg_join_mask
);

  // -------------------------------------------------------------------
  // All-valid check for masked inputs
  // -------------------------------------------------------------------
  logic all_selected_valid;

  always_comb begin : check_valid
    integer iter_var0;
    all_selected_valid = 1'b1;
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : check_each
      if (cfg_join_mask[iter_var0] && !in_valid[iter_var0])
        all_selected_valid = 1'b0;
    end : check_each
  end : check_valid

  // Output valid when all selected inputs present.
  assign out_valid = all_selected_valid;

  // Output data is none-type: always zero.
  assign out_data = '0;

  // Transfer indicator
  logic out_transfer;
  assign out_transfer = out_valid & out_ready;

  // -------------------------------------------------------------------
  // Input ready logic
  // -------------------------------------------------------------------
  always_comb begin : ready_logic
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : set_ready
      if (cfg_join_mask[iter_var0])
        in_ready[iter_var0] = out_transfer;
      else
        in_ready[iter_var0] = 1'b1;  // Non-selected: always accept/discard
    end : set_ready
  end : ready_logic

endmodule : fu_op_join
