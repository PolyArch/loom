// Dataflow invariant: store value 'a' once, then output it for each 'd' token.
module dataflow_invariant #(
    parameter int WIDTH = 32
) (
    input  logic             clk,
    input  logic             rst_n,

    // Done signal input
    input  logic             d_valid,
    output logic             d_ready,

    // Value to store
    input  logic             a_valid,
    output logic             a_ready,
    input  logic [WIDTH-1:0] a_data,

    // Output (repeated value)
    output logic             o_valid,
    input  logic             o_ready,
    output logic [WIDTH-1:0] o_data
);

  typedef enum logic [1:0] {
    S_LOAD   = 2'b00,
    S_REPEAT = 2'b01
  } state_t;

  state_t state;
  logic [WIDTH-1:0] stored_value;

  always_ff @(posedge clk or negedge rst_n) begin : fsm
    if (!rst_n) begin : reset
      state        <= S_LOAD;
      stored_value <= '0;
    end else begin : operate
      case (state)
        S_LOAD: begin : load_state
          if (a_valid) begin : load_fire
            stored_value <= a_data;
            state        <= S_REPEAT;
          end
        end
        S_REPEAT: begin : repeat_state
          // Stay in REPEAT, output stored value for each d token
        end
        default: begin : default_state
          state <= S_LOAD;
        end
      endcase
    end
  end

  always_comb begin : output_logic
    case (state)
      S_LOAD: begin : load_out
        o_valid = 1'b0;
        o_data  = '0;
        a_ready = 1'b1;
        d_ready = 1'b0;
      end
      S_REPEAT: begin : repeat_out
        o_valid = d_valid;
        o_data  = stored_value;
        a_ready = 1'b0;
        d_ready = o_ready;
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
