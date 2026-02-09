// Dataflow gate: stream alignment by cutting head of condition and tail of value.
// afterCond[i] = beforeCond[i+1], afterValue[i] = beforeValue[i]
// Implements a one-token delay on the condition stream relative to the value.
module dataflow_gate #(
    parameter int WIDTH = 32
) (
    input  logic             clk,
    input  logic             rst_n,

    // Before-value input
    input  logic             bv_valid,
    output logic             bv_ready,
    input  logic [WIDTH-1:0] bv_data,

    // Before-condition input
    input  logic             bc_valid,
    output logic             bc_ready,
    input  logic             bc_data,

    // After-value output
    output logic             av_valid,
    input  logic             av_ready,
    output logic [WIDTH-1:0] av_data,

    // After-condition output
    output logic             ac_valid,
    input  logic             ac_ready,
    output logic             ac_data
);

  typedef enum logic [1:0] {
    S_SKIP_HEAD = 2'b00,
    S_FORWARD   = 2'b01
  } state_t;

  state_t state;

  always_ff @(posedge clk or negedge rst_n) begin : fsm
    if (!rst_n) begin : reset
      state <= S_SKIP_HEAD;
    end else begin : operate
      case (state)
        S_SKIP_HEAD: begin : skip_state
          // Consume first condition token (discard it)
          if (bc_valid) begin : skip_fire
            state <= S_FORWARD;
          end
        end
        S_FORWARD: begin : forward_state
          // Stay in forward mode
        end
        default: begin : default_state
          state <= S_SKIP_HEAD;
        end
      endcase
    end
  end

  always_comb begin : output_logic
    case (state)
      S_SKIP_HEAD: begin : skip_out
        // Consume first condition, block everything else
        bc_ready = 1'b1;
        bv_ready = 1'b0;
        av_valid = 1'b0;
        av_data  = '0;
        ac_valid = 1'b0;
        ac_data  = 1'b0;
      end
      S_FORWARD: begin : forward_out
        // Forward value and shifted condition together
        av_valid = bv_valid && bc_valid;
        av_data  = bv_data;
        ac_valid = bv_valid && bc_valid;
        ac_data  = bc_data;
        bv_ready = av_ready && ac_ready && bc_valid;
        bc_ready = av_ready && ac_ready && bv_valid;
      end
      default: begin : default_out
        bc_ready = 1'b0;
        bv_ready = 1'b0;
        av_valid = 1'b0;
        av_data  = '0;
        ac_valid = 1'b0;
        ac_data  = 1'b0;
      end
    endcase
  end

endmodule
