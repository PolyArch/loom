// tb_backpressure_test.sv -- Verify data integrity under random backpressure.
//
// Instantiates a fabric_fifo (depth=4) and applies tb_backpressure_gen on
// the output side. Sends NUM_TOKENS sequential values through the FIFO
// and verifies that all tokens arrive in order despite random stalls.
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

module tb_backpressure_test;

    // =========================================================================
    // Parameters
    // =========================================================================
    parameter DATA_WIDTH         = 32;
    parameter FIFO_DEPTH         = 4;
    parameter NUM_TOKENS         = 64;
    parameter CLK_PERIOD_NS      = 10;
    parameter RST_CYCLES         = 5;
    parameter SIM_TIMEOUT_CYCLES = 10000;
    parameter BP_SEED            = 12345;
    parameter BP_PROB_READY_PCT  = 60;

    // =========================================================================
    // Clock and reset
    // =========================================================================
    wire clk;
    wire rst_n;

    tb_clk_rst_gen #(
        .CLK_PERIOD_NS (CLK_PERIOD_NS),
        .RST_CYCLES    (RST_CYCLES)
    ) u_clk_rst (
        .clk   (clk),
        .rst_n (rst_n)
    );

    // =========================================================================
    // Backpressure generator on the output side
    // =========================================================================
    wire bp_ready;

    tb_backpressure_gen #(
        .SEED           (BP_SEED),
        .PROB_READY_PCT (BP_PROB_READY_PCT)
    ) u_bp (
        .clk    (clk),
        .rst_n  (rst_n),
        .enable (1'b1),
        .ready  (bp_ready)
    );

    // =========================================================================
    // DUT: fabric_fifo (depth=4, non-bypassable, no tag)
    // =========================================================================
    logic                   in_valid;
    logic                   in_ready;
    logic [DATA_WIDTH-1:0]  in_data;

    logic                   out_valid;
    logic                   out_ready;
    logic [DATA_WIDTH-1:0]  out_data;

    fabric_fifo #(
        .DEPTH      (FIFO_DEPTH),
        .DATA_WIDTH (DATA_WIDTH),
        .TAG_WIDTH  (0),
        .BYPASSABLE (1'b0)
    ) u_dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .cfg_valid (1'b0),
        .cfg_wdata (32'd0),
        .cfg_ready (),
        .in_valid  (in_valid),
        .in_ready  (in_ready),
        .in_data   (in_data),
        .in_tag    (1'b0),
        .out_valid (out_valid),
        .out_ready (out_ready),
        .out_data  (out_data),
        .out_tag   ()
    );

    // Output ready driven by backpressure generator
    assign out_ready = bp_ready;

    // =========================================================================
    // Input driver: sends sequential values 0..NUM_TOKENS-1
    // =========================================================================
    integer send_idx;
    logic   send_done;

    // Drive in_data combinationally from send_idx so the FIFO sees the
    // correct value on the same cycle that in_valid is high.
    always_comb begin : driver_comb
        in_data = send_idx[DATA_WIDTH-1:0];
    end

    always @(posedge clk or negedge rst_n) begin : driver_proc
        if (!rst_n) begin : driver_reset
            send_idx  <= 0;
            in_valid  <= 1'b0;
            send_done <= 1'b0;
        end else begin : driver_active
            if (send_done) begin : driver_done_hold
                in_valid <= 1'b0;
            end else begin : driver_send
                in_valid <= 1'b1;

                if (in_valid && in_ready) begin : driver_advance
                    if (send_idx + 1 >= NUM_TOKENS) begin : driver_last
                        send_done <= 1'b1;
                        in_valid  <= 1'b0;
                    end else begin : driver_next
                        send_idx <= send_idx + 1;
                    end
                end
            end
        end
    end

    // =========================================================================
    // Output checker: verifies tokens arrive in order
    // =========================================================================
    integer recv_idx;
    integer mismatch_count;
    logic   recv_done;

    always @(posedge clk or negedge rst_n) begin : checker_proc
        if (!rst_n) begin : checker_reset
            recv_idx       <= 0;
            mismatch_count <= 0;
            recv_done      <= 1'b0;
        end else begin : checker_active
            if (out_valid && out_ready) begin : checker_transfer
                if (out_data !== recv_idx[DATA_WIDTH-1:0]) begin : checker_mismatch
                    if (mismatch_count < 10) begin : checker_report
                        $display("[tb_backpressure_test] MISMATCH[%0d]: got %h, expected %h",
                                 recv_idx, out_data, recv_idx[DATA_WIDTH-1:0]);
                    end
                    mismatch_count <= mismatch_count + 1;
                end

                if (recv_idx + 1 >= NUM_TOKENS) begin : checker_last
                    recv_done <= 1'b1;
                end else begin : checker_next
                    recv_idx <= recv_idx + 1;
                end
            end
        end
    end

    // =========================================================================
    // Simulation timeout and verdict
    // =========================================================================
    integer cycle_count;

    always @(posedge clk or negedge rst_n) begin : timeout_proc
        if (!rst_n) begin : timeout_reset
            cycle_count <= 0;
        end else begin : timeout_active
            cycle_count <= cycle_count + 1;

            if (recv_done) begin : verdict_check
                if (mismatch_count == 0) begin : verdict_pass
                    $display("[tb_backpressure_test] PASS: All %0d tokens received correctly under backpressure",
                             NUM_TOKENS);
                end else begin : verdict_fail
                    $display("[tb_backpressure_test] FAIL: %0d mismatches out of %0d tokens",
                             mismatch_count, NUM_TOKENS);
                end
                $finish;
            end

            if (cycle_count >= SIM_TIMEOUT_CYCLES) begin : timeout_hit
                $display("[tb_backpressure_test] FAIL: Timeout at cycle %0d (sent %0d, received %0d)",
                         cycle_count, send_idx, recv_idx);
                $finish;
            end
        end
    end

endmodule
