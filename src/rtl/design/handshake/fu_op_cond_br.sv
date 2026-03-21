// fu_op_cond_br.sv -- Conditional branch (handshake.cond_br).
//
// Routes data to one of two outputs based on a boolean condition.
// Combinational: intrinsic latency 0.
//
// Inputs:
//   0: cond (i1, LSB selects output)
//   1: data (any, WIDTH bits)
//
// Outputs:
//   0: true_out  (any, WIDTH bits) -- receives data when cond=1
//   1: false_out (any, WIDTH bits) -- receives data when cond=0
//
// Handshake contract:
//   - Both inputs must be valid to fire.
//   - Only the selected output asserts valid.
//   - Both inputs are consumed when the selected output completes transfer.
//   - The non-selected output remains invalid (no backpressure from it).

module fu_op_cond_br #(
  parameter int unsigned WIDTH = 32
) (
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic                clk,
  input  logic                rst_n,
  /* verilator lint_on UNUSEDSIGNAL */

  // Input 0: cond (only LSB used)
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [WIDTH-1:0]    in_data_0,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic                in_valid_0,
  output logic                in_ready_0,

  // Input 1: data
  input  logic [WIDTH-1:0]    in_data_1,
  input  logic                in_valid_1,
  output logic                in_ready_1,

  // Output 0: true_out
  output logic [WIDTH-1:0]    out_data_0,
  output logic                out_valid_0,
  input  logic                out_ready_0,

  // Output 1: false_out
  output logic [WIDTH-1:0]    out_data_1,
  output logic                out_valid_1,
  input  logic                out_ready_1
);

  // Both inputs valid
  logic both_valid;
  assign both_valid = in_valid_0 & in_valid_1;

  // Selection based on LSB of cond
  logic sel;
  assign sel = in_data_0[0];

  // Output valid: only the selected output
  assign out_valid_0 = both_valid &  sel;
  assign out_valid_1 = both_valid & ~sel;

  // Data passes through to both outputs (only the valid one matters)
  assign out_data_0 = in_data_1;
  assign out_data_1 = in_data_1;

  // Ready to the selected output
  logic selected_ready;
  assign selected_ready = sel ? out_ready_0 : out_ready_1;

  // Inputs are consumed when both are valid and the selected output accepts
  assign in_ready_0 = both_valid & selected_ready;
  assign in_ready_1 = both_valid & selected_ready;

endmodule : fu_op_cond_br
