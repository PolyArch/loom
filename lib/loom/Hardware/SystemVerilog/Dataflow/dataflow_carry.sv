// Dataflow carry: outputs initial value 'a' first, then 'b' repeatedly.
// State machine: INIT -> output a, then CARRY -> output b on each 'd' token.
module dataflow_carry #(
    parameter int WIDTH = 32
) (
    input  logic             clk,
    input  logic             rst_n,

    // Done signal input
    input  logic             d_valid,
    output logic             d_ready,

    // Initial value input
    input  logic             a_valid,
    output logic             a_ready,
    input  logic [WIDTH-1:0] a_data,

    // Carry value input
    input  logic             b_valid,
    output logic             b_ready,
    input  logic [WIDTH-1:0] b_data,

    // Output
    output logic             o_valid,
    input  logic             o_ready,
    output logic [WIDTH-1:0] o_data
);

  typedef enum logic [1:0] {
    S_INIT  = 2'b00,
    S_CARRY = 2'b01
  } state_t;

  state_t state;

  always_ff @(posedge clk or negedge rst_n) begin : fsm
    if (!rst_n) begin : reset
      state <= S_INIT;
    end else begin : operate
      case (state)
        S_INIT: begin : init_state
          if (a_valid && o_ready) begin : init_fire
            state <= S_CARRY;
          end
        end
        S_CARRY: begin : carry_state
          if (d_valid && b_valid && o_ready) begin : carry_fire
            // Stay in CARRY until done
          end
        end
        default: begin : default_state
          state <= S_INIT;
        end
      endcase
    end
  end

  // Output mux based on state
  always_comb begin : output_mux
    case (state)
      S_INIT: begin : init_out
        o_valid = a_valid;
        o_data  = a_data;
        a_ready = o_ready;
        b_ready = 1'b0;
        d_ready = 1'b0;
      end
      S_CARRY: begin : carry_out
        o_valid = d_valid && b_valid;
        o_data  = b_data;
        a_ready = 1'b0;
        b_ready = d_valid && o_ready;
        d_ready = b_valid && o_ready;
      end
      default: begin : default_out
        o_valid = 1'b0;
        o_data  = '0;
        a_ready = 1'b0;
        b_ready = 1'b0;
        d_ready = 1'b0;
      end
    endcase
  end

endmodule
