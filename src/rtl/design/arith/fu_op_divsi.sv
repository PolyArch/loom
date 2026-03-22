// fu_op_divsi.sv -- Signed integer division FU operation.
//
// Multi-cycle iterative divider using restoring division.
// result = signed(a) / signed(b), or 0 when b == 0.
// Intrinsic latency: WIDTH+2 cycles.
//
// Protocol: When both inputs are valid and the unit is idle, it
// captures operands, asserts busy, and begins computation. After
// WIDTH+2 cycles the result appears on out_data with out_valid high.
// The unit returns to idle when the output is transferred (out_ready).

module fu_op_divsi #(
  parameter int unsigned WIDTH = 32
) (
  input  logic                clk,
  input  logic                rst_n,

  // Input operand A (dividend)
  input  logic [WIDTH-1:0]    in_data_0,
  input  logic                in_valid_0,
  output logic                in_ready_0,

  // Input operand B (divisor)
  input  logic [WIDTH-1:0]    in_data_1,
  input  logic                in_valid_1,
  output logic                in_ready_1,

  // Output result (quotient)
  output logic [WIDTH-1:0]    out_data,
  output logic                out_valid,
  input  logic                out_ready
);

  // State encoding.
  typedef enum logic [1:0] {
    ST_IDLE    = 2'd0,
    ST_COMPUTE = 2'd1,
    ST_DONE    = 2'd2
  } state_t;

  state_t state_r, state_next;

  // Iteration counter.
  localparam int unsigned CNT_WIDTH = $clog2(WIDTH + 1);
  logic [CNT_WIDTH-1:0] cnt_r, cnt_next;

  // Internal registers for the restoring division algorithm.
  logic [WIDTH-1:0]   quotient_r,  quotient_next;
  logic [WIDTH:0]     remainder_r, remainder_next;
  logic [WIDTH-1:0]   divisor_r,   divisor_next;
  logic               negate_q_r,  negate_q_next;

  // Input handshake: accept when idle and both valid.
  logic inputs_valid;
  assign inputs_valid = in_valid_0 & in_valid_1;
  assign in_ready_0   = (state_r == ST_IDLE) & inputs_valid;
  assign in_ready_1   = (state_r == ST_IDLE) & inputs_valid;

  // Output handshake.
  assign out_valid = (state_r == ST_DONE);

  // Absolute value computation for signed division.
  logic [WIDTH-1:0] abs_a;
  logic [WIDTH-1:0] abs_b;
  logic              a_neg;
  logic              b_neg;

  always_comb begin : abs_compute
    a_neg = in_data_0[WIDTH-1];
    b_neg = in_data_1[WIDTH-1];
    abs_a = a_neg ? (~in_data_0 + {{(WIDTH-1){1'b0}}, 1'b1}) : in_data_0;
    abs_b = b_neg ? (~in_data_1 + {{(WIDTH-1){1'b0}}, 1'b1}) : in_data_1;
  end : abs_compute

  // Next-state logic.
  always_comb begin : next_state_logic
    state_next     = state_r;
    cnt_next       = cnt_r;
    quotient_next  = quotient_r;
    remainder_next = remainder_r;
    divisor_next   = divisor_r;
    negate_q_next  = negate_q_r;

    case (state_r)
      ST_IDLE: begin : idle_case
        if (inputs_valid) begin : start_div
          if (in_data_1 == {WIDTH{1'b0}}) begin : div_by_zero
            // Division by zero: result is 0.
            quotient_next  = {WIDTH{1'b0}};
            state_next     = ST_DONE;
          end : div_by_zero
          else begin : start_compute
            quotient_next  = {WIDTH{1'b0}};
            remainder_next = {(WIDTH + 1){1'b0}};
            divisor_next   = abs_b;
            negate_q_next  = a_neg ^ b_neg;
            // Load the MSB of the absolute dividend into remainder LSB
            // via shift in first iteration. Store abs_a in quotient_r
            // as a shift source.
            quotient_next  = abs_a;
            cnt_next       = {CNT_WIDTH{1'b0}};
            state_next     = ST_COMPUTE;
          end : start_compute
        end : start_div
      end : idle_case

      ST_COMPUTE: begin : compute_case
        // Restoring division: shift remainder left, bring in next
        // dividend bit from quotient_r MSB.
        logic [WIDTH:0] trial_sub;
        logic [WIDTH:0] shifted_rem;

        shifted_rem = {remainder_r[WIDTH-1:0], quotient_r[WIDTH-1]};
        trial_sub   = shifted_rem - {1'b0, divisor_r};

        if (!trial_sub[WIDTH]) begin : sub_positive
          // Subtraction succeeded.
          remainder_next = trial_sub;
          quotient_next  = {quotient_r[WIDTH-2:0], 1'b1};
        end : sub_positive
        else begin : sub_negative
          // Restore: keep shifted remainder.
          remainder_next = shifted_rem;
          quotient_next  = {quotient_r[WIDTH-2:0], 1'b0};
        end : sub_negative

        if (cnt_r == CNT_WIDTH'(WIDTH - 1)) begin : last_iter
          state_next = ST_DONE;
        end : last_iter
        else begin : more_iters
          cnt_next = cnt_r + {{(CNT_WIDTH-1){1'b0}}, 1'b1};
        end : more_iters
      end : compute_case

      ST_DONE: begin : done_case
        if (out_ready) begin : transfer_out
          state_next = ST_IDLE;
        end : transfer_out
      end : done_case

      default: begin : default_case
        state_next = ST_IDLE;
      end : default_case
    endcase
  end : next_state_logic

  // Registered state update.
  always_ff @(posedge clk) begin : state_update
    if (!rst_n) begin : reset_block
      state_r     <= ST_IDLE;
      cnt_r       <= {CNT_WIDTH{1'b0}};
      quotient_r  <= {WIDTH{1'b0}};
      remainder_r <= {(WIDTH + 1){1'b0}};
      divisor_r   <= {WIDTH{1'b0}};
      negate_q_r  <= 1'b0;
    end : reset_block
    else begin : normal_block
      state_r     <= state_next;
      cnt_r       <= cnt_next;
      quotient_r  <= quotient_next;
      remainder_r <= remainder_next;
      divisor_r   <= divisor_next;
      negate_q_r  <= negate_q_next;
    end : normal_block
  end : state_update

  // Output: negate quotient if signs differ.
  assign out_data = negate_q_r
                    ? (~quotient_r + {{(WIDTH-1){1'b0}}, 1'b1})
                    : quotient_r;

endmodule : fu_op_divsi
