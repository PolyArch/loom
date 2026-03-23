// tb_noc_traffic_gen.sv -- Configurable traffic generator for NoC testing.
//
// Generates flit traffic with selectable patterns: uniform random,
// hotspot (all traffic to a single destination), and all-to-all
// (round-robin across destinations).
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

module tb_noc_traffic_gen
  import noc_pkg::*;
#(
  parameter int unsigned DATA_WIDTH    = NOC_DATA_WIDTH_DEFAULT,
  parameter int unsigned MESH_ROWS     = 2,
  parameter int unsigned MESH_COLS     = 2,
  parameter int unsigned SRC_ID        = 0,
  parameter int unsigned NUM_FLITS     = 16,
  parameter int unsigned INJECT_RATE   = 50,   // percentage (0-100)
  // Traffic pattern: 0=uniform random, 1=hotspot, 2=all-to-all
  parameter int unsigned PATTERN       = 0,
  parameter int unsigned HOTSPOT_DST   = 0
)(
  input  logic clk,
  input  logic rst_n,

  // Injection interface.
  output logic [flit_width(DATA_WIDTH)-1:0] flit_out,
  output logic                              flit_valid,
  input  logic                              flit_ready,

  // Status.
  output logic        done,
  output logic [31:0] sent_count
);

  // ---------------------------------------------------------------
  // Localparams
  // ---------------------------------------------------------------
  localparam int unsigned FLIT_W      = flit_width(DATA_WIDTH);
  localparam int unsigned NUM_ROUTERS = MESH_ROWS * MESH_COLS;

  // ---------------------------------------------------------------
  // Internal state
  // ---------------------------------------------------------------
  logic [31:0] flit_idx;
  logic [31:0] rng_state;
  logic [NOC_ID_WIDTH-1:0] dst_counter;

  // ---------------------------------------------------------------
  // Simple LFSR-based pseudo-random number generator
  // ---------------------------------------------------------------
  function automatic logic [31:0] lfsr_next(input logic [31:0] state);
    logic feedback;
    feedback = state[31] ^ state[21] ^ state[1] ^ state[0];
    return {state[30:0], feedback};
  endfunction : lfsr_next

  // ---------------------------------------------------------------
  // Destination selection based on pattern
  // ---------------------------------------------------------------
  logic [NOC_ID_WIDTH-1:0] next_dst;

  always_comb begin : dst_select
    case (PATTERN)
      1: begin : pattern_hotspot
        next_dst = NOC_ID_WIDTH'(HOTSPOT_DST);
      end : pattern_hotspot
      2: begin : pattern_all_to_all
        next_dst = dst_counter;
      end : pattern_all_to_all
      default: begin : pattern_random
        // Use rng_state to select a destination (excluding self).
        next_dst = NOC_ID_WIDTH'(rng_state % NUM_ROUTERS);
      end : pattern_random
    endcase
  end : dst_select

  // ---------------------------------------------------------------
  // Flit construction
  // ---------------------------------------------------------------
  flit_header_t out_header;
  logic [DATA_WIDTH-1:0] out_payload;

  assign out_header.flit_type = FLIT_SINGLE;
  assign out_header.src_id    = NOC_ID_WIDTH'(SRC_ID);
  assign out_header.dst_id    = next_dst;
  assign out_header.vc_id     = '0;
  assign out_payload          = flit_idx[DATA_WIDTH-1:0];
  assign flit_out             = {out_header, out_payload};

  // ---------------------------------------------------------------
  // Injection control
  // ---------------------------------------------------------------
  logic inject_enable;
  assign inject_enable = (rng_state[7:0] < 8'(INJECT_RATE * 255 / 100));

  always_ff @(posedge clk or negedge rst_n) begin : gen_proc
    if (!rst_n) begin : gen_reset
      flit_idx   <= '0;
      rng_state  <= 32'hDEAD_BEEF + SRC_ID[31:0];
      dst_counter <= '0;
      done       <= 1'b0;
      sent_count <= '0;
      flit_valid <= 1'b0;
    end : gen_reset
    else begin : gen_active
      rng_state <= lfsr_next(rng_state);

      if (done) begin : gen_done_hold
        flit_valid <= 1'b0;
      end : gen_done_hold
      else begin : gen_sending
        if (flit_valid && flit_ready) begin : gen_accepted
          sent_count <= sent_count + 1;
          flit_idx   <= flit_idx + 1;
          flit_valid <= 1'b0;

          // Advance all-to-all destination counter.
          if (PATTERN == 2) begin : gen_aa_advance
            if (dst_counter == NOC_ID_WIDTH'(NUM_ROUTERS - 1)) begin : gen_aa_wrap
              dst_counter <= '0;
            end : gen_aa_wrap
            else begin : gen_aa_incr
              dst_counter <= dst_counter + 1'b1;
            end : gen_aa_incr
          end : gen_aa_advance

          if (flit_idx + 1 >= NUM_FLITS[31:0]) begin : gen_finish
            done <= 1'b1;
          end : gen_finish
        end : gen_accepted
        else if (!flit_valid && inject_enable) begin : gen_inject
          flit_valid <= 1'b1;
        end : gen_inject
      end : gen_sending
    end : gen_active
  end : gen_proc

endmodule : tb_noc_traffic_gen
