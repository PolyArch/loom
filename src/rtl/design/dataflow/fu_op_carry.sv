// fu_op_carry.sv -- Loop-carried dependency state machine (dataflow.carry).
//
// Dedicated dataflow state-machine FU.  latency=-1, interval=-1.
//
// Inputs:
//   0: d (i1, loop-condition, LSB only)
//   1: a (any, initial value)
//   2: b (any, loop-carried value)
//
// Output:
//   0: o (any, carried output)
//
// State machine (matching spec-dataflow.md and simulator commitCarry):
//   NEED_INIT (initial) - accept a on input 1, emit on output, go to WAIT_INIT_ACK
//   WAIT_INIT_ACK       - wait for output accepted, go to NEED_COND
//   NEED_COND           - accept d on input 0
//       d=true  -> go to NEED_LOOP
//       d=false -> go back to NEED_INIT
//   NEED_LOOP           - accept b on input 2, emit on output, go to WAIT_LOOP_ACK
//   WAIT_LOOP_ACK       - wait for output accepted, go to NEED_COND

module fu_op_carry #(
  parameter int unsigned WIDTH = 32
) (
  input  logic                clk,
  input  logic                rst_n,

  // Input 0: d (loop condition, LSB is boolean)
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [WIDTH-1:0]    in_data_0,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic                in_valid_0,
  output logic                in_ready_0,

  // Input 1: a (initial value)
  input  logic [WIDTH-1:0]    in_data_1,
  input  logic                in_valid_1,
  output logic                in_ready_1,

  // Input 2: b (loop-carried value)
  input  logic [WIDTH-1:0]    in_data_2,
  input  logic                in_valid_2,
  output logic                in_ready_2,

  // Output 0: o (carried output)
  output logic [WIDTH-1:0]    out_data,
  output logic                out_valid,
  input  logic                out_ready
);

  // -------------------------------------------------------------------
  // State encoding
  // -------------------------------------------------------------------
  typedef enum logic [2:0] {
    ST_NEED_INIT      = 3'd0,
    ST_WAIT_INIT_ACK  = 3'd1,
    ST_NEED_COND      = 3'd2,
    ST_NEED_LOOP      = 3'd3,
    ST_WAIT_LOOP_ACK  = 3'd4
  } state_t;

  state_t state_r;

  // -------------------------------------------------------------------
  // Output holding register
  // -------------------------------------------------------------------
  logic              out_valid_r;
  logic [WIDTH-1:0]  out_data_r;

  assign out_valid = out_valid_r;
  assign out_data  = out_data_r;

  logic out_transfer;
  assign out_transfer = out_valid & out_ready;

  // -------------------------------------------------------------------
  // Input ready logic
  // -------------------------------------------------------------------
  always_comb begin : input_ready_logic
    in_ready_0 = 1'b0;
    in_ready_1 = 1'b0;
    in_ready_2 = 1'b0;
    case (state_r)
      ST_NEED_INIT: begin : init_ready
        in_ready_1 = 1'b1;
      end : init_ready
      ST_NEED_COND: begin : cond_ready
        in_ready_0 = 1'b1;
      end : cond_ready
      ST_NEED_LOOP: begin : loop_ready
        in_ready_2 = 1'b1;
      end : loop_ready
      default: ;
    endcase
  end : input_ready_logic

  // -------------------------------------------------------------------
  // Main sequential logic
  // -------------------------------------------------------------------
  always_ff @(posedge clk) begin : main_seq
    if (!rst_n) begin : reset_block
      state_r    <= ST_NEED_INIT;
      out_valid_r <= 1'b0;
      out_data_r  <= '0;
    end : reset_block
    else begin : active_block
      case (state_r)
        // ---------------------------------------------------------
        ST_NEED_INIT: begin : state_need_init
          if (in_valid_1) begin : fire_init
            out_valid_r <= 1'b1;
            out_data_r  <= in_data_1;
            state_r     <= ST_WAIT_INIT_ACK;
          end : fire_init
        end : state_need_init

        // ---------------------------------------------------------
        ST_WAIT_INIT_ACK: begin : state_wait_init_ack
          if (out_transfer) begin : init_ack
            out_valid_r <= 1'b0;
            state_r     <= ST_NEED_COND;
          end : init_ack
        end : state_wait_init_ack

        // ---------------------------------------------------------
        ST_NEED_COND: begin : state_need_cond
          if (in_valid_0) begin : fire_cond
            if (in_data_0[0]) begin : cond_true
              state_r <= ST_NEED_LOOP;
            end : cond_true
            else begin : cond_false
              state_r <= ST_NEED_INIT;
            end : cond_false
          end : fire_cond
        end : state_need_cond

        // ---------------------------------------------------------
        ST_NEED_LOOP: begin : state_need_loop
          if (in_valid_2) begin : fire_loop
            out_valid_r <= 1'b1;
            out_data_r  <= in_data_2;
            state_r     <= ST_WAIT_LOOP_ACK;
          end : fire_loop
        end : state_need_loop

        // ---------------------------------------------------------
        ST_WAIT_LOOP_ACK: begin : state_wait_loop_ack
          if (out_transfer) begin : loop_ack
            out_valid_r <= 1'b0;
            state_r     <= ST_NEED_COND;
          end : loop_ack
        end : state_wait_loop_ack

        default: begin : state_default
          state_r <= ST_NEED_INIT;
        end : state_default
      endcase
    end : active_block
  end : main_seq

endmodule : fu_op_carry
