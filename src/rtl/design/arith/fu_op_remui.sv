// fu_op_remui.sv -- Unsigned integer remainder FU operation.
//
// Multi-cycle iterative divider using restoring division.
// result = a % b (unsigned), or 0 when b == 0.
// Intrinsic latency: WIDTH+2 cycles.

module fu_op_remui #(
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

  logic inputs_valid;
  assign inputs_valid = in_valid_0 & in_valid_1;
  assign in_ready_0   = (state_r == ST_IDLE) & inputs_valid;
  assign in_ready_1   = (state_r == ST_IDLE) & inputs_valid;

  assign out_valid = (state_r == ST_DONE);
  assign out_data  = remainder_r[WIDTH-1:0];

  always_comb begin : next_state_logic
    state_next     = state_r;
    cnt_next       = cnt_r;
    quotient_next  = quotient_r;
    remainder_next = remainder_r;
    divisor_next   = divisor_r;

    case (state_r)
      ST_IDLE: begin : idle_case
        if (inputs_valid) begin : start_div
          if (in_data_1 == {WIDTH{1'b0}}) begin : div_by_zero
            quotient_next  = {WIDTH{1'b0}};
            remainder_next = {(WIDTH + 1){1'b0}};
            state_next     = ST_DONE;
          end : div_by_zero
          else begin : start_compute
            quotient_next  = in_data_0;
            remainder_next = {(WIDTH + 1){1'b0}};
            divisor_next   = in_data_1;
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
    end : reset_block
    else begin : normal_block
      state_r     <= state_next;
      cnt_r       <= cnt_next;
      quotient_r  <= quotient_next;
      remainder_r <= remainder_next;
      divisor_r   <= divisor_next;
    end : normal_block
  end : state_update

endmodule : fu_op_remui
