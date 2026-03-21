// fu_op_gate.sv -- Before-to-after stream adapter (dataflow.gate).
//
// Dedicated dataflow state-machine FU.  latency=-1, interval=-1.
//
// Inputs:
//   0: beforeValue (any, WIDTH bits)
//   1: beforeCond  (i1, carried on WIDTH bits, LSB only)
//
// Outputs:
//   0: afterValue (any, WIDTH bits)
//   1: afterCond  (i1, carried on WIDTH bits, LSB only)
//
// State machine (matching simulator commitGate / evaluateGate):
//   NEED_HEAD (state 0) - consume (value, cond) pair
//       cond=true  -> emit afterValue only, go to WAIT_HEAD_ACK
//       cond=false -> discard, stay in NEED_HEAD
//   WAIT_HEAD_ACK (state 1) - wait for afterValue accepted
//       -> go to NEED_NEXT
//   NEED_NEXT (state 2) - consume next (value, cond) pair
//       cond=true  -> emit afterValue + afterCond=1, go to WAIT_BOTH_ACK
//       cond=false -> emit afterCond=0 only, go to WAIT_COND_ACK
//   WAIT_BOTH_ACK (state 3) - wait for both outputs accepted
//       -> go to NEED_NEXT
//   WAIT_COND_ACK (state 4) - wait for afterCond accepted
//       -> go to NEED_HEAD

module fu_op_gate #(
  parameter int unsigned WIDTH = 32
) (
  input  logic                clk,
  input  logic                rst_n,

  // Input 0: beforeValue
  input  logic [WIDTH-1:0]    in_data_0,
  input  logic                in_valid_0,
  output logic                in_ready_0,

  // Input 1: beforeCond (LSB is the boolean)
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [WIDTH-1:0]    in_data_1,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic                in_valid_1,
  output logic                in_ready_1,

  // Output 0: afterValue
  output logic [WIDTH-1:0]    out_data_0,
  output logic                out_valid_0,
  input  logic                out_ready_0,

  // Output 1: afterCond (LSB is the boolean)
  output logic [WIDTH-1:0]    out_data_1,
  output logic                out_valid_1,
  input  logic                out_ready_1
);

  // -------------------------------------------------------------------
  // State encoding
  // -------------------------------------------------------------------
  typedef enum logic [2:0] {
    ST_NEED_HEAD      = 3'd0,
    ST_WAIT_HEAD_ACK  = 3'd1,
    ST_NEED_NEXT      = 3'd2,
    ST_WAIT_BOTH_ACK  = 3'd3,
    ST_WAIT_COND_ACK  = 3'd4
  } state_t;

  state_t state_r;

  // -------------------------------------------------------------------
  // Input capture registers
  // -------------------------------------------------------------------
  logic              value_captured_r;
  logic [WIDTH-1:0]  value_data_r;
  logic              cond_captured_r;
  logic              cond_val_r;

  // -------------------------------------------------------------------
  // Output holding registers
  // -------------------------------------------------------------------
  logic              out0_valid_r;
  logic [WIDTH-1:0]  out0_data_r;
  logic              out1_valid_r;
  logic [WIDTH-1:0]  out1_data_r;

  // Per-output accepted tracking for atomic broadcast
  logic              out0_accepted_r;
  logic              out1_accepted_r;

  // -------------------------------------------------------------------
  // Output drive
  // -------------------------------------------------------------------
  assign out_valid_0 = out0_valid_r;
  assign out_data_0  = out0_data_r;
  assign out_valid_1 = out1_valid_r;
  assign out_data_1  = out1_data_r;

  logic out0_transfer;
  logic out1_transfer;
  assign out0_transfer = out_valid_0 & out_ready_0;
  assign out1_transfer = out_valid_1 & out_ready_1;

  // -------------------------------------------------------------------
  // Input ready logic
  // -------------------------------------------------------------------
  always_comb begin : input_ready_logic
    in_ready_0 = 1'b0;
    in_ready_1 = 1'b0;
    case (state_r)
      ST_NEED_HEAD: begin : head_ready
        in_ready_0 = ~value_captured_r;
        in_ready_1 = ~cond_captured_r;
      end : head_ready
      ST_NEED_NEXT: begin : next_ready
        in_ready_0 = ~value_captured_r;
        in_ready_1 = ~cond_captured_r;
      end : next_ready
      default: ;
    endcase
  end : input_ready_logic

  // -------------------------------------------------------------------
  // Main sequential logic
  // -------------------------------------------------------------------
  always_ff @(posedge clk) begin : main_seq
    if (!rst_n) begin : reset_block
      state_r          <= ST_NEED_HEAD;
      value_captured_r <= 1'b0;
      value_data_r     <= '0;
      cond_captured_r  <= 1'b0;
      cond_val_r       <= 1'b0;
      out0_valid_r     <= 1'b0;
      out0_data_r      <= '0;
      out1_valid_r     <= 1'b0;
      out1_data_r      <= '0;
      out0_accepted_r  <= 1'b0;
      out1_accepted_r  <= 1'b0;
    end : reset_block
    else begin : active_block

      // Track per-output acceptance
      if (out0_transfer)
        out0_accepted_r <= 1'b1;
      if (out1_transfer)
        out1_accepted_r <= 1'b1;

      case (state_r)
        // ---------------------------------------------------------
        ST_NEED_HEAD: begin : state_need_head
          // Capture inputs independently
          if (in_valid_0 && !value_captured_r) begin : cap_val_head
            value_data_r     <= in_data_0;
            value_captured_r <= 1'b1;
          end : cap_val_head

          if (in_valid_1 && !cond_captured_r) begin : cap_cond_head
            cond_val_r      <= in_data_1[0];
            cond_captured_r <= 1'b1;
          end : cap_cond_head

          // Both captured (including this cycle captures)
          if ((value_captured_r || (in_valid_0 && !value_captured_r)) &&
              (cond_captured_r  || (in_valid_1 && !cond_captured_r))) begin : fire_head
            // Determine cond value (use latched or incoming)
            if ((cond_captured_r ? cond_val_r : in_data_1[0])) begin : head_true
              // Emit afterValue only
              out0_valid_r <= 1'b1;
              out0_data_r  <= value_captured_r ? value_data_r : in_data_0;
              out0_accepted_r <= 1'b0;
              state_r      <= ST_WAIT_HEAD_ACK;
            end : head_true
            // else: cond=false, discard, stay in NEED_HEAD

            // Clear capture flags
            value_captured_r <= 1'b0;
            cond_captured_r  <= 1'b0;
          end : fire_head
        end : state_need_head

        // ---------------------------------------------------------
        ST_WAIT_HEAD_ACK: begin : state_wait_head_ack
          if (out0_accepted_r || out0_transfer) begin : head_ack
            out0_valid_r    <= 1'b0;
            out0_accepted_r <= 1'b0;
            state_r         <= ST_NEED_NEXT;
          end : head_ack
        end : state_wait_head_ack

        // ---------------------------------------------------------
        ST_NEED_NEXT: begin : state_need_next
          // Capture inputs independently
          if (in_valid_0 && !value_captured_r) begin : cap_val_next
            value_data_r     <= in_data_0;
            value_captured_r <= 1'b1;
          end : cap_val_next

          if (in_valid_1 && !cond_captured_r) begin : cap_cond_next
            cond_val_r      <= in_data_1[0];
            cond_captured_r <= 1'b1;
          end : cap_cond_next

          // Both captured
          if ((value_captured_r || (in_valid_0 && !value_captured_r)) &&
              (cond_captured_r  || (in_valid_1 && !cond_captured_r))) begin : fire_next
            if ((cond_captured_r ? cond_val_r : in_data_1[0])) begin : next_true
              // Emit afterValue and afterCond=1
              out0_valid_r <= 1'b1;
              out0_data_r  <= value_captured_r ? value_data_r : in_data_0;
              out1_valid_r <= 1'b1;
              out1_data_r  <= {{(WIDTH-1){1'b0}}, 1'b1};
              out0_accepted_r <= 1'b0;
              out1_accepted_r <= 1'b0;
              state_r      <= ST_WAIT_BOTH_ACK;
            end : next_true
            else begin : next_false
              // Emit afterCond=0 only
              out1_valid_r <= 1'b1;
              out1_data_r  <= '0;
              out1_accepted_r <= 1'b0;
              state_r      <= ST_WAIT_COND_ACK;
            end : next_false

            value_captured_r <= 1'b0;
            cond_captured_r  <= 1'b0;
          end : fire_next
        end : state_need_next

        // ---------------------------------------------------------
        ST_WAIT_BOTH_ACK: begin : state_wait_both_ack
          if ((out0_accepted_r || out0_transfer) &&
              (out1_accepted_r || out1_transfer)) begin : both_ack
            out0_valid_r    <= 1'b0;
            out1_valid_r    <= 1'b0;
            out0_accepted_r <= 1'b0;
            out1_accepted_r <= 1'b0;
            state_r         <= ST_NEED_NEXT;
          end : both_ack
        end : state_wait_both_ack

        // ---------------------------------------------------------
        ST_WAIT_COND_ACK: begin : state_wait_cond_ack
          if (out1_accepted_r || out1_transfer) begin : cond_ack
            out1_valid_r    <= 1'b0;
            out1_accepted_r <= 1'b0;
            state_r         <= ST_NEED_HEAD;
          end : cond_ack
        end : state_wait_cond_ack

        default: begin : state_default
          state_r <= ST_NEED_HEAD;
        end : state_default
      endcase
    end : active_block
  end : main_seq

endmodule : fu_op_gate
