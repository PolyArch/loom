//===-- fabric_double_buffer.sv - 2-entry skid buffer -----------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Two-entry elastic buffer for decoupled valid/ready streams. Breaks
// combinational paths between producer ready and consumer valid while
// sustaining full throughput (one transfer per cycle in steady state).
//
// Ready is determined solely by buffer occupancy -- no combinational
// dependency on the input valid signal.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module fabric_double_buffer #(
    parameter int WIDTH = 1,
    localparam int SAFE_W = (WIDTH > 0) ? WIDTH : 1
) (
    input  logic              clk,
    input  logic              rst_n,

    // Input (enqueue) side
    input  logic              in_valid,
    output logic              in_ready,
    input  logic [SAFE_W-1:0] in_data,

    // Output (dequeue) side
    output logic              out_valid,
    input  logic              out_ready,
    output logic [SAFE_W-1:0] out_data
);

  logic v0, v1;
  logic [SAFE_W-1:0] d0, d1;

  logic enq_fire, deq_fire;
  assign enq_fire = in_valid && in_ready;
  assign deq_fire = out_valid && out_ready;

  // Ready when second slot is empty (can always absorb one more)
  assign in_ready  = !v1;
  // Valid when first slot is occupied
  assign out_valid = v0;
  assign out_data  = d0;

  always_ff @(posedge clk or negedge rst_n) begin : state_update
    if (!rst_n) begin : reset_block
      v0 <= 1'b0;
      v1 <= 1'b0;
      d0 <= '0;
      d1 <= '0;
    end else begin : normal_block
      case ({enq_fire, deq_fire})
        2'b00: begin : no_op
        end
        2'b01: begin : deq_only
          if (v1) begin : shift_down
            d0 <= d1;
            v1 <= 1'b0;
          end else begin : drain
            v0 <= 1'b0;
          end
        end
        2'b10: begin : enq_only
          if (!v0) begin : fill_first
            v0 <= 1'b1;
            d0 <= in_data;
          end else begin : fill_second
            v1 <= 1'b1;
            d1 <= in_data;
          end
        end
        2'b11: begin : simultaneous
          if (v1) begin : shift_and_fill
            d0 <= d1;
            d1 <= in_data;
          end else begin : replace_head
            d0 <= in_data;
          end
        end
      endcase
    end
  end

endmodule
