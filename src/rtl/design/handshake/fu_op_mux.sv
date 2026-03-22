// fu_op_mux.sv -- Runtime selector mux (handshake.mux).
//
// Combinational: intrinsic latency 0.
//
// This is the runtime dataflow mux (handshake.mux), NOT the
// configuration-time structural selector (fabric.mux).
//
// Inputs:
//   0:            sel    (index, selects which data input to consume)
//   1..NUM_DATA:  data_i (any, WIDTH bits each)
//
// Output:
//   0: result (any, WIDTH bits)
//
// Handshake contract:
//   - sel is consumed first.
//   - sel.data selects which data input (0-indexed) to consume.
//   - Only the selected data input is consumed; non-selected inputs
//     are blocked (ready=0) and their tokens are NOT consumed.
//   - Output fires when sel and the selected data input are both valid.
//
// The sel value maps to data input index: sel=0 -> data_0, sel=1 -> data_1, etc.

module fu_op_mux #(
  parameter int unsigned NUM_DATA  = 2,   // Number of data inputs (not counting sel)
  parameter int unsigned WIDTH     = 32
) (
  input  logic                clk,
  input  logic                rst_n,

  // Input 0: sel (only lower bits used for index)
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [WIDTH-1:0]    in_data_sel,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic                in_valid_sel,
  output logic                in_ready_sel,

  // Data inputs: NUM_DATA ports
  input  logic [NUM_DATA-1:0][WIDTH-1:0] in_data,
  input  logic [NUM_DATA-1:0]            in_valid,
  output logic [NUM_DATA-1:0]            in_ready,

  // Output: result
  output logic [WIDTH-1:0]    out_data,
  output logic                out_valid,
  input  logic                out_ready
);

  // -------------------------------------------------------------------
  // Selector index width
  // -------------------------------------------------------------------
  localparam int unsigned SEL_WIDTH = (NUM_DATA > 1) ? $clog2(NUM_DATA) : 1;

  // -------------------------------------------------------------------
  // Selector capture register
  // -------------------------------------------------------------------
  logic                   sel_captured_r;
  logic [SEL_WIDTH-1:0]   sel_val_r;

  // Effective selector index (from capture reg or live input)
  logic [SEL_WIDTH-1:0] sel_idx;
  assign sel_idx = sel_captured_r ? sel_val_r : in_data_sel[SEL_WIDTH-1:0];

  // Selector available (either captured or currently valid)
  logic sel_available;
  assign sel_available = sel_captured_r | in_valid_sel;

  // Range check: is sel_idx pointing to a valid data port?
  logic sel_in_range;
  generate
    if (NUM_DATA == (1 << SEL_WIDTH)) begin : full_range_gen
      // When NUM_DATA is exactly a power of 2, index is always in range.
      assign sel_in_range = 1'b1;
    end : full_range_gen
    else begin : partial_range_gen
      assign sel_in_range = ({1'b0, sel_idx} < (SEL_WIDTH+1)'(NUM_DATA));
    end : partial_range_gen
  endgenerate

  // -------------------------------------------------------------------
  // Selected data valid check
  // -------------------------------------------------------------------
  logic selected_valid;

  always_comb begin : sel_valid_check
    selected_valid = 1'b0;
    if (sel_available && sel_in_range)
      selected_valid = in_valid[sel_idx];
  end : sel_valid_check

  // -------------------------------------------------------------------
  // Output logic
  // -------------------------------------------------------------------
  assign out_valid = sel_available & selected_valid;

  always_comb begin : sel_data_mux
    out_data = '0;
    if (sel_available && sel_in_range)
      out_data = in_data[sel_idx];
  end : sel_data_mux

  logic out_transfer;
  assign out_transfer = out_valid & out_ready;

  // -------------------------------------------------------------------
  // Input ready logic
  // -------------------------------------------------------------------

  // sel ready: accept sel when not yet captured and either we can
  // complete a transfer now or sel is not yet available
  assign in_ready_sel = ~sel_captured_r & (out_transfer | ~sel_available);

  // Data input ready: only the selected input, only on transfer
  always_comb begin : data_ready_logic
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_DATA; iter_var0 = iter_var0 + 1) begin : set_data_ready
      if (sel_available && sel_in_range && (iter_var0[SEL_WIDTH-1:0] == sel_idx))
        in_ready[iter_var0] = out_transfer;
      else
        in_ready[iter_var0] = 1'b0;
    end : set_data_ready
  end : data_ready_logic

  // -------------------------------------------------------------------
  // Selector capture FSM
  // -------------------------------------------------------------------
  always_ff @(posedge clk) begin : sel_capture_seq
    if (!rst_n) begin : reset_block
      sel_captured_r <= 1'b0;
      sel_val_r      <= '0;
    end : reset_block
    else begin : active_block
      if (out_transfer) begin : clear_sel
        sel_captured_r <= 1'b0;
      end : clear_sel
      else if (!sel_captured_r && in_valid_sel) begin : capture_sel
        sel_captured_r <= 1'b1;
        sel_val_r      <= in_data_sel[SEL_WIDTH-1:0];
      end : capture_sel
    end : active_block
  end : sel_capture_seq

endmodule : fu_op_mux
