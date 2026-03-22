// tb_backpressure_gen.sv -- Randomized ready signal generator for
//                           backpressure testing.
//
// Produces a pseudorandom 'ready' signal based on a configurable
// probability. Useful for stress-testing handshake modules under
// varying backpressure conditions.
//
// When 'enable' is low, ready is forced high (no backpressure).
// When 'enable' is high, ready toggles pseudorandomly each cycle
// with approximately PROB_READY_PCT percent probability of being high.
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

module tb_backpressure_gen #(
    parameter SEED           = 42,
    parameter PROB_READY_PCT = 80   // 0-100: probability of ready being high
)(
    input  wire clk,
    input  wire rst_n,
    input  wire enable,

    output reg  ready
);

    // -------------------------------------------------------------------------
    // Random state -- uses blocking assignments intentionally (testbench-only
    // random number computation, not synthesizable state).
    // -------------------------------------------------------------------------
    /* verilator lint_off BLKSEQ */
    integer rand_val;
    integer seed_state;

    initial begin : seed_init
        seed_state = SEED;
    end

    // -------------------------------------------------------------------------
    // Ready generation
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin : gen_proc
        if (!rst_n) begin : gen_reset
            ready <= 1'b1;
        end else begin : gen_active
            if (!enable) begin : gen_disabled
                ready <= 1'b1;
            end else begin : gen_enabled
                rand_val = $urandom(seed_state) % 100;
                seed_state = seed_state + 1;
                if (rand_val < PROB_READY_PCT) begin : gen_ready_high
                    ready <= 1'b1;
                end else begin : gen_ready_low
                    ready <= 1'b0;
                end
            end
        end
    end
    /* verilator lint_on BLKSEQ */

endmodule
