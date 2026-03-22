// fu_op_stream.sv -- Configurable index-stream generator (dataflow.stream).
//
// Dedicated dataflow state-machine FU.  latency=-1, interval=-1.
//
// Inputs:
//   0: start (index)
//   1: step  (index)
//   2: bound (index)
//
// Outputs:
//   0: idx          (index)
//   1: willContinue (i1, carried on WIDTH bits, LSB only)
//
// Config: cont_cond  -- 5-bit one-hot comparison selector
//           bit 0: <    bit 1: <=   bit 2: >    bit 3: >=   bit 4: !=
//
// State machine:
//   IDLE    -> capture start/step/bound independently -> INIT
//   INIT    -> evaluate first (idx, cont), emit, advance -> RUNNING
//   RUNNING -> both outputs accepted -> emit next or -> TERMINAL
//   TERMINAL -> both outputs accepted -> IDLE

module fu_op_stream #(
  parameter int unsigned WIDTH    = 32,
  parameter int unsigned CFG_WIDTH = 5   // cont_cond one-hot width
) (
  input  logic                clk,
  input  logic                rst_n,

  // Input 0: start
  input  logic [WIDTH-1:0]    in_data_0,
  input  logic                in_valid_0,
  output logic                in_ready_0,

  // Input 1: step
  input  logic [WIDTH-1:0]    in_data_1,
  input  logic                in_valid_1,
  output logic                in_ready_1,

  // Input 2: bound
  input  logic [WIDTH-1:0]    in_data_2,
  input  logic                in_valid_2,
  output logic                in_ready_2,

  // Output 0: idx
  output logic [WIDTH-1:0]    out_data_0,
  output logic                out_valid_0,
  input  logic                out_ready_0,

  // Output 1: willContinue (1-bit value in WIDTH-bit container)
  output logic [WIDTH-1:0]    out_data_1,
  output logic                out_valid_1,
  input  logic                out_ready_1,

  // Configuration: cont_cond
  input  logic [CFG_WIDTH-1:0] cfg_cont_cond
);

  // -------------------------------------------------------------------
  // State encoding
  // -------------------------------------------------------------------
  typedef enum logic [1:0] {
    ST_IDLE     = 2'd0,
    ST_INIT     = 2'd1,
    ST_RUNNING  = 2'd2,
    ST_TERMINAL = 2'd3
  } state_t;

  state_t state_r;

  // -------------------------------------------------------------------
  // Captured input registers
  // -------------------------------------------------------------------
  logic              start_captured_r;
  logic              step_captured_r;
  logic              bound_captured_r;
  logic [WIDTH-1:0]  start_val_r;
  logic [WIDTH-1:0]  step_val_r;
  logic [WIDTH-1:0]  bound_val_r;

  // Running state registers
  logic [WIDTH-1:0]  next_idx_r;
  logic [WIDTH-1:0]  step_reg_r;
  logic [WIDTH-1:0]  bound_reg_r;
  // (terminal state is encoded in state_r == ST_TERMINAL)

  // Output holding registers
  logic              out0_valid_r;
  logic [WIDTH-1:0]  out0_data_r;
  logic              out1_valid_r;
  logic [WIDTH-1:0]  out1_data_r;

  // Output accepted tracking (for atomic broadcast of both outputs)
  logic              out0_accepted_r;
  logic              out1_accepted_r;

  // -------------------------------------------------------------------
  // Continuation condition evaluator (signed comparison)
  // -------------------------------------------------------------------
  logic cont_result;

  always_comb begin : eval_cont_cond
    cont_result = 1'b0;
    if (cfg_cont_cond[0])
      cont_result = ($signed(next_idx_r) < $signed(bound_reg_r));
    else if (cfg_cont_cond[1])
      cont_result = ($signed(next_idx_r) <= $signed(bound_reg_r));
    else if (cfg_cont_cond[2])
      cont_result = ($signed(next_idx_r) > $signed(bound_reg_r));
    else if (cfg_cont_cond[3])
      cont_result = ($signed(next_idx_r) >= $signed(bound_reg_r));
    else if (cfg_cont_cond[4])
      cont_result = (next_idx_r != bound_reg_r);
  end : eval_cont_cond

  // Same evaluator but for the init phase where we use start_val_r
  logic init_cont_result;

  always_comb begin : eval_init_cont_cond
    init_cont_result = 1'b0;
    if (cfg_cont_cond[0])
      init_cont_result = ($signed(start_val_r) < $signed(bound_val_r));
    else if (cfg_cont_cond[1])
      init_cont_result = ($signed(start_val_r) <= $signed(bound_val_r));
    else if (cfg_cont_cond[2])
      init_cont_result = ($signed(start_val_r) > $signed(bound_val_r));
    else if (cfg_cont_cond[3])
      init_cont_result = ($signed(start_val_r) >= $signed(bound_val_r));
    else if (cfg_cont_cond[4])
      init_cont_result = (start_val_r != bound_val_r);
  end : eval_init_cont_cond

  // -------------------------------------------------------------------
  // Input ready logic
  // -------------------------------------------------------------------
  always_comb begin : input_ready_logic
    in_ready_0 = 1'b0;
    in_ready_1 = 1'b0;
    in_ready_2 = 1'b0;
    if (state_r == ST_IDLE) begin : idle_ready
      in_ready_0 = ~start_captured_r;
      in_ready_1 = ~step_captured_r;
      in_ready_2 = ~bound_captured_r;
    end : idle_ready
  end : input_ready_logic

  // -------------------------------------------------------------------
  // Output drive logic
  // -------------------------------------------------------------------
  assign out_valid_0 = out0_valid_r;
  assign out_data_0  = out0_data_r;
  assign out_valid_1 = out1_valid_r;
  assign out_data_1  = out1_data_r;

  // Transfer indicators
  logic out0_transfer;
  logic out1_transfer;
  assign out0_transfer = out_valid_0 & out_ready_0;
  assign out1_transfer = out_valid_1 & out_ready_1;

  // -------------------------------------------------------------------
  // Main sequential logic
  // -------------------------------------------------------------------
  always_ff @(posedge clk) begin : main_seq
    if (!rst_n) begin : reset_block
      state_r            <= ST_IDLE;
      start_captured_r   <= 1'b0;
      step_captured_r    <= 1'b0;
      bound_captured_r   <= 1'b0;
      start_val_r        <= '0;
      step_val_r         <= '0;
      bound_val_r        <= '0;
      next_idx_r         <= '0;
      step_reg_r         <= '0;
      bound_reg_r        <= '0;
      out0_valid_r       <= 1'b0;
      out0_data_r        <= '0;
      out1_valid_r       <= 1'b0;
      out1_data_r        <= '0;
      out0_accepted_r    <= 1'b0;
      out1_accepted_r    <= 1'b0;
    end : reset_block
    else begin : active_block

      // Track per-output acceptance
      if (out0_transfer)
        out0_accepted_r <= 1'b1;
      if (out1_transfer)
        out1_accepted_r <= 1'b1;

      case (state_r)
        // ---------------------------------------------------------
        ST_IDLE: begin : state_idle
          // Independently capture start, step, bound
          if (in_valid_0 && !start_captured_r) begin : cap_start
            start_val_r      <= in_data_0;
            start_captured_r <= 1'b1;
          end : cap_start

          if (in_valid_1 && !step_captured_r) begin : cap_step
            step_val_r      <= in_data_1;
            step_captured_r <= 1'b1;
          end : cap_step

          if (in_valid_2 && !bound_captured_r) begin : cap_bound
            bound_val_r      <= in_data_2;
            bound_captured_r <= 1'b1;
          end : cap_bound

          // All three captured: check combinationally (including
          // captures happening this cycle)
          if ((start_captured_r || (in_valid_0 && !start_captured_r)) &&
              (step_captured_r  || (in_valid_1 && !step_captured_r))  &&
              (bound_captured_r || (in_valid_2 && !bound_captured_r))) begin : all_captured
            state_r <= ST_INIT;
          end : all_captured
        end : state_idle

        // ---------------------------------------------------------
        ST_INIT: begin : state_init
          // Emit first (idx, cont) pair
          out0_valid_r       <= 1'b1;
          out0_data_r        <= start_val_r;
          out1_valid_r       <= 1'b1;
          out1_data_r        <= {{(WIDTH-1){1'b0}}, init_cont_result};
          out0_accepted_r    <= 1'b0;
          out1_accepted_r    <= 1'b0;

          // Set up running state
          next_idx_r         <= start_val_r + step_val_r;
          step_reg_r         <= step_val_r;
          bound_reg_r        <= bound_val_r;

          // Clear capture flags
          start_captured_r   <= 1'b0;
          step_captured_r    <= 1'b0;
          bound_captured_r   <= 1'b0;

          if (init_cont_result)
            state_r <= ST_RUNNING;
          else
            state_r <= ST_TERMINAL;
        end : state_init

        // ---------------------------------------------------------
        ST_RUNNING: begin : state_running
          // Wait for both outputs to be accepted
          if ((out0_accepted_r || out0_transfer) &&
              (out1_accepted_r || out1_transfer)) begin : both_accepted
            // Clear output registers
            out0_accepted_r <= 1'b0;
            out1_accepted_r <= 1'b0;

            // Emit next value
            out0_valid_r <= 1'b1;
            out0_data_r  <= next_idx_r;
            out1_valid_r <= 1'b1;
            out1_data_r  <= {{(WIDTH-1){1'b0}}, cont_result};

            if (cont_result) begin : continue_emit
              next_idx_r <= next_idx_r + step_reg_r;
            end : continue_emit
            else begin : terminal_emit
              state_r <= ST_TERMINAL;
            end : terminal_emit
          end : both_accepted
        end : state_running

        // ---------------------------------------------------------
        ST_TERMINAL: begin : state_terminal
          // Wait for both outputs to be accepted, then return to idle
          if ((out0_accepted_r || out0_transfer) &&
              (out1_accepted_r || out1_transfer)) begin : term_accepted
            out0_valid_r    <= 1'b0;
            out1_valid_r    <= 1'b0;
            out0_accepted_r <= 1'b0;
            out1_accepted_r <= 1'b0;
            state_r         <= ST_IDLE;
          end : term_accepted
        end : state_terminal

        default: begin : state_default
          state_r <= ST_IDLE;
        end : state_default
      endcase
    end : active_block
  end : main_seq

endmodule : fu_op_stream
