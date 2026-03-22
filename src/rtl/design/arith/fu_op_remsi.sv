// fu_op_remsi.sv -- Signed integer remainder FU operation.
//
// Multi-cycle iterative divider using restoring division.
// result = signed(a) % signed(b), or 0 when b == 0.
// The sign of the remainder matches the sign of the dividend (C semantics).
// Intrinsic latency: WIDTH+2 cycles.

module fu_op_remsi #(
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

  // Output result (remainder)
  output logic [WIDTH-1:0]    out_data,
  output logic                out_valid,
  input  logic                out_ready
);

  typedef enum logic [1:0] {
    ST_IDLE    = 2'd0,
    ST_COMPUTE = 2'd1,
    ST_DONE    = 2'd2
  } state_t;

  state_t state_r, state_next;

  localparam int unsigned CNT_WIDTH = $clog2(WIDTH + 1);
  logic [CNT_WIDTH-1:0] cnt_r, cnt_next;

  logic [WIDTH-1:0]   quotient_r,  quotient_next;
  logic [WIDTH:0]     remainder_r, remainder_next;
  logic [WIDTH-1:0]   divisor_r,   divisor_next;
  logic               negate_r_r,  negate_r_next;

  logic inputs_valid;
  assign inputs_valid = in_valid_0 & in_valid_1;
  assign in_ready_0   = (state_r == ST_IDLE) & inputs_valid;
  assign in_ready_1   = (state_r == ST_IDLE) & inputs_valid;

  assign out_valid = (state_r == ST_DONE);

  // Absolute value computation.
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

  always_comb begin : next_state_logic
    state_next     = state_r;
    cnt_next       = cnt_r;
    quotient_next  = quotient_r;
    remainder_next = remainder_r;
    divisor_next   = divisor_r;
    negate_r_next  = negate_r_r;

    case (state_r)
      ST_IDLE: begin : idle_case
        if (inputs_valid) begin : start_div
          if (in_data_1 == {WIDTH{1'b0}}) begin : div_by_zero
            quotient_next  = {WIDTH{1'b0}};
            remainder_next = {(WIDTH + 1){1'b0}};
            state_next     = ST_DONE;
          end : div_by_zero
          else begin : start_compute
            quotient_next  = abs_a;
            remainder_next = {(WIDTH + 1){1'b0}};
            divisor_next   = abs_b;
            negate_r_next  = a_neg;
            cnt_next       = {CNT_WIDTH{1'b0}};
            state_next     = ST_COMPUTE;
          end : start_compute
        end : start_div
      end : idle_case

      ST_COMPUTE: begin : compute_case
        logic [WIDTH:0] trial_sub;
        logic [WIDTH:0] shifted_rem;

        shifted_rem = {remainder_r[WIDTH-1:0], quotient_r[WIDTH-1]};
        trial_sub   = shifted_rem - {1'b0, divisor_r};

        if (!trial_sub[WIDTH]) begin : sub_positive
          remainder_next = trial_sub;
          quotient_next  = {quotient_r[WIDTH-2:0], 1'b1};
        end : sub_positive
        else begin : sub_negative
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

  always_ff @(posedge clk) begin : state_update
    if (!rst_n) begin : reset_block
      state_r     <= ST_IDLE;
      cnt_r       <= {CNT_WIDTH{1'b0}};
      quotient_r  <= {WIDTH{1'b0}};
      remainder_r <= {(WIDTH + 1){1'b0}};
      divisor_r   <= {WIDTH{1'b0}};
      negate_r_r  <= 1'b0;
    end : reset_block
    else begin : normal_block
      state_r     <= state_next;
      cnt_r       <= cnt_next;
      quotient_r  <= quotient_next;
      remainder_r <= remainder_next;
      divisor_r   <= divisor_next;
      negate_r_r  <= negate_r_next;
    end : normal_block
  end : state_update

  // Output: the remainder is in remainder_r[WIDTH-1:0], negated if
  // the dividend was negative (C signed-remainder semantics).
  logic [WIDTH-1:0] abs_remainder;
  assign abs_remainder = remainder_r[WIDTH-1:0];

  assign out_data = negate_r_r
                    ? (~abs_remainder + {{(WIDTH-1){1'b0}}, 1'b1})
                    : abs_remainder;

endmodule : fu_op_remsi
