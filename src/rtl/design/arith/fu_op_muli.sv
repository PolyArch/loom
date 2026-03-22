// fu_op_muli.sv -- Integer multiply FU operation.
//
// Uses combinational multiply with ready/valid handshake.
// The intrinsic latency is 0 in this implementation (purely combinational).
// The PE slot wrapper adds retiming registers if the declared latency > 0.
//
// For synthesis, the synthesis tool will infer a multiplier from the
// behavioral description. DesignWare or other IP can be substituted
// via the slot wrapper or synthesis constraints.
//
// Intrinsic latency: 0 (combinational; slot wrapper adds pipeline stages).

module fu_op_muli #(
  parameter int unsigned WIDTH = 32
) (
  // verilator lint_off UNUSEDSIGNAL
  input  logic                clk,
  input  logic                rst_n,
  // verilator lint_on UNUSEDSIGNAL

  // Input operand A
  input  logic [WIDTH-1:0]    in_data_0,
  input  logic                in_valid_0,
  output logic                in_ready_0,

  // Input operand B
  input  logic [WIDTH-1:0]    in_data_1,
  input  logic                in_valid_1,
  output logic                in_ready_1,

  // Output result (lower WIDTH bits of the product)
  output logic [WIDTH-1:0]    out_data,
  output logic                out_valid,
  input  logic                out_ready
);

  assign out_valid = in_valid_0 & in_valid_1;

  assign in_ready_0 = out_ready & out_valid;
  assign in_ready_1 = out_ready & out_valid;

  // Full product is 2*WIDTH bits; we take only the lower WIDTH bits,
  // matching the simulator behavior (C uint64_t multiplication wraps).
  // verilator lint_off UNUSEDSIGNAL
  logic [2*WIDTH-1:0] full_product;
  // verilator lint_on UNUSEDSIGNAL
  assign full_product = in_data_0 * in_data_1;
  assign out_data     = full_product[WIDTH-1:0];

endmodule : fu_op_muli
