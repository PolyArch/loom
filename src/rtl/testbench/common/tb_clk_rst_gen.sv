// tb_clk_rst_gen.sv -- Clock and active-low reset generator for testbenches
//
// Generates a free-running clock and an active-low reset that deasserts
// after RST_CYCLES clock cycles. Non-synthesizable (testbench only).

`timescale 1ns/1ps

module tb_clk_rst_gen #(
    parameter CLK_PERIOD_NS = 10,
    parameter RST_CYCLES    = 5
)(
    output reg clk,
    output reg rst_n
);

    // Half period for toggling
    localparam HALF_PERIOD = CLK_PERIOD_NS / 2.0;

    // -------------------------------------------------------------------------
    // Clock generation
    // -------------------------------------------------------------------------
    initial begin : clk_init
        clk = 1'b0;
        forever begin : clk_toggle
            #(HALF_PERIOD) clk = ~clk;
        end
    end

    // -------------------------------------------------------------------------
    // Reset generation -- rst_n asserted (low) for RST_CYCLES then released
    // -------------------------------------------------------------------------
    initial begin : rst_init
        integer iter_var0;
        rst_n = 1'b0;
        for (iter_var0 = 0; iter_var0 < RST_CYCLES; iter_var0 = iter_var0 + 1) begin : rst_wait_loop
            @(posedge clk);
        end
        @(negedge clk);
        rst_n = 1'b1;
    end

endmodule
