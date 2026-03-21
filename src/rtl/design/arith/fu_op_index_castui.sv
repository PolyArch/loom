// fu_op_index_castui.sv -- Width adaptation (zero-extend or truncate).
//
// Combinational: adapts IN_WIDTH to OUT_WIDTH.
//   - When OUT_WIDTH > IN_WIDTH: zero-extends.
//   - When OUT_WIDTH < IN_WIDTH: truncates (takes lower bits).
//   - When OUT_WIDTH == IN_WIDTH: passthrough.
// Intrinsic latency: 0.

module fu_op_index_castui #(
  parameter int unsigned IN_WIDTH  = 32,
  parameter int unsigned OUT_WIDTH = 32
) (
  // verilator lint_off UNUSEDSIGNAL
  input  logic                  clk,
  input  logic                  rst_n,
  // verilator lint_on UNUSEDSIGNAL

  // Input operand A (upper bits may be unused when truncating)
  // verilator lint_off UNUSEDSIGNAL
  input  logic [IN_WIDTH-1:0]   in_data_0,
  // verilator lint_on UNUSEDSIGNAL
  input  logic                  in_valid_0,
  output logic                  in_ready_0,

  // Output result
  output logic [OUT_WIDTH-1:0]  out_data,
  output logic                  out_valid,
  input  logic                  out_ready
);

  assign out_valid  = in_valid_0;
  assign in_ready_0 = out_ready & out_valid;

  generate
    if (OUT_WIDTH > IN_WIDTH) begin : gen_zero_extend
      // Zero-extend: pad upper bits with zeros.
      assign out_data = {{(OUT_WIDTH - IN_WIDTH){1'b0}}, in_data_0};
    end : gen_zero_extend
    else if (OUT_WIDTH < IN_WIDTH) begin : gen_truncate
      // Truncate: take lower bits.
      assign out_data = in_data_0[OUT_WIDTH-1:0];
    end : gen_truncate
    else begin : gen_passthrough
      assign out_data = in_data_0;
    end : gen_passthrough
  endgenerate

endmodule : fu_op_index_castui
