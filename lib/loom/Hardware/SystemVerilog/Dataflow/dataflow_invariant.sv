// Dataflow invariant: emit initial value 'a' on output, store it internally,
// then repeat it for each 'd' token where d_data=1. When d_data=0, consume
// only d and return to S_INIT for a new value.
module dataflow_invariant #(
    parameter int WIDTH = 32
) (
    input  logic             clk,
    input  logic             rst_n,

    // Done signal input
    input  logic             d_valid,
    output logic             d_ready,
    input  logic             d_data,

    // Value to store
    input  logic             a_valid,
    output logic             a_ready,
    input  logic [WIDTH-1:0] a_data,

    // Output (repeated value)
    output logic             o_valid,
    input  logic             o_ready,
    output logic [WIDTH-1:0] o_data
);

  typedef enum logic {
    S_INIT  = 1'b0,
    S_BLOCK = 1'b1
  } state_t;

  state_t state;
  logic [WIDTH-1:0] stored_value;

  always_ff @(posedge clk or negedge rst_n) begin : fsm
    if (!rst_n) begin : reset
      state        <= S_INIT;
      stored_value <= '0;
    end else begin : operate
      case (state)
        S_INIT: begin : init_state
          if (a_valid && o_ready) begin : init_fire
            stored_value <= a_data;
            state        <= S_BLOCK;
          end
        end
        S_BLOCK: begin : block_state
          if (d_valid && !d_data) begin : block_done
            // d_data=0: consume only d, return to S_INIT
            state <= S_INIT;
          end
        end
        default: begin : default_state
          state <= S_INIT;
        end
      endcase
    end
  end

  always_comb begin : output_logic
    case (state)
      S_INIT: begin : init_out
        o_valid = a_valid;
        o_data  = a_data;
        a_ready = o_ready;
        d_ready = 1'b0;
      end
      S_BLOCK: begin : block_out
        if (d_data) begin : block_continue
          o_valid = d_valid;
          o_data  = stored_value;
          a_ready = 1'b0;
          d_ready = o_ready;
        end else begin : block_finish
          o_valid = 1'b0;
          o_data  = '0;
          a_ready = 1'b0;
          d_ready = 1'b1;
        end
      end
      default: begin : default_out
        o_valid = 1'b0;
        o_data  = '0;
        a_ready = 1'b0;
        d_ready = 1'b0;
      end
    endcase
  end

endmodule
