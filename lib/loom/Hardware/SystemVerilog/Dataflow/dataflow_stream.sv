// Dataflow stream: loop counter with configurable continuation condition.
// Generates index values from start with step, bounded by bound.
// cfg_cont_cond_sel selects the comparison: one-hot encoding of
// {slt, sle, sgt, sge, ne} (bits 0-4).
//
// STEP_OP selects the step operation: 0=+=, 1=-=, 2=*=, 3=/=, 4=<<=, 5=>>=.
//
// Errors:
//   CFG_PE_STREAM_CONT_COND_ONEHOT - cont_cond_sel is not one-hot
//   RT_DATAFLOW_STREAM_ZERO_STEP   - step == 0 at runtime

`include "fabric_common.svh"

module dataflow_stream #(
    parameter int WIDTH = 32,
    parameter int STEP_OP = 0
) (
    input  logic             clk,
    input  logic             rst_n,

    // Start value
    input  logic             start_valid,
    output logic             start_ready,
    input  logic [WIDTH-1:0] start_data,

    // Step value
    input  logic             step_valid,
    output logic             step_ready,
    input  logic [WIDTH-1:0] step_data,

    // Bound value
    input  logic             bound_valid,
    output logic             bound_ready,
    input  logic [WIDTH-1:0] bound_data,

    // Index output
    output logic             index_valid,
    input  logic             index_ready,
    output logic [WIDTH-1:0] index_data,

    // Will-continue flag output
    output logic             cont_valid,
    input  logic             cont_ready,
    output logic             cont_data,

    // Configuration: continuation condition selector (one-hot)
    input  logic [4:0]       cfg_cont_cond_sel,

    // Error output
    output logic             error_valid,
    output logic [15:0]      error_code
);

  typedef enum logic [1:0] {
    S_IDLE    = 2'b00,
    S_RUNNING = 2'b01
  } state_t;

  state_t state;
  logic [WIDTH-1:0] current_index;
  logic [WIDTH-1:0] saved_step;
  logic [WIDTH-1:0] saved_bound;

  // Continuation condition evaluation (uses current_index, not next_index)
  logic will_continue;
  logic [WIDTH-1:0] next_index;

  // Step operation mux
  always_comb begin : step_op_mux
    case (STEP_OP)
      0: next_index = current_index + saved_step;
      1: next_index = current_index - saved_step;
      2: next_index = current_index * saved_step;
      3: next_index = current_index / saved_step;
      4: next_index = current_index << saved_step;
      5: next_index = current_index >> saved_step;
      default: next_index = current_index + saved_step;
    endcase
  end

  always_comb begin : eval_cont
    will_continue = 1'b0;
    case (1'b1)
      cfg_cont_cond_sel[0]: will_continue = $signed(current_index) <  $signed(saved_bound); // slt
      cfg_cont_cond_sel[1]: will_continue = $signed(current_index) <= $signed(saved_bound); // sle
      cfg_cont_cond_sel[2]: will_continue = $signed(current_index) >  $signed(saved_bound); // sgt
      cfg_cont_cond_sel[3]: will_continue = $signed(current_index) >= $signed(saved_bound); // sge
      cfg_cont_cond_sel[4]: will_continue = (current_index != saved_bound);                  // ne
      default: will_continue = 1'b0;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin : fsm
    if (!rst_n) begin : reset
      state         <= S_IDLE;
      current_index <= '0;
      saved_step    <= '0;
      saved_bound   <= '0;
    end else begin : operate
      case (state)
        S_IDLE: begin : idle_state
          if (start_valid && step_valid && bound_valid) begin : init_fire
            current_index <= start_data;
            saved_step    <= step_data;
            saved_bound   <= bound_data;
            state         <= S_RUNNING;
          end
        end
        S_RUNNING: begin : running_state
          if (index_ready && cont_ready) begin : advance
            if (will_continue) begin : next_iter
              current_index <= next_index;
            end else begin : done
              state <= S_IDLE;
            end
          end
        end
        default: begin : default_state
          state <= S_IDLE;
        end
      endcase
    end
  end

  // Output signals
  assign index_valid = (state == S_RUNNING);
  assign index_data  = current_index;
  assign cont_valid  = (state == S_RUNNING);
  assign cont_data   = will_continue;

  // Input handshake
  assign start_ready = (state == S_IDLE) && step_valid && bound_valid;
  assign step_ready  = (state == S_IDLE) && start_valid && bound_valid;
  assign bound_ready = (state == S_IDLE) && start_valid && step_valid;

  // Error detection
  logic cfg_not_onehot;
  logic rt_zero_step;

  // One-hot check: exactly one bit set
  assign cfg_not_onehot = (cfg_cont_cond_sel == '0) ||
                          (cfg_cont_cond_sel & (cfg_cont_cond_sel - 1)) != '0;
  assign rt_zero_step   = (state == S_RUNNING) && (saved_step == '0);

  always_ff @(posedge clk or negedge rst_n) begin : error_latch
    if (!rst_n) begin : reset_err
      error_valid <= 1'b0;
      error_code  <= 16'd0;
    end else if (!error_valid) begin : capture
      if (cfg_not_onehot) begin : onehot_err
        error_valid <= 1'b1;
        error_code  <= CFG_PE_STREAM_CONT_COND_ONEHOT;
      end else if (rt_zero_step) begin : step_err
        error_valid <= 1'b1;
        error_code  <= RT_DATAFLOW_STREAM_ZERO_STEP;
      end
    end
  end

endmodule
