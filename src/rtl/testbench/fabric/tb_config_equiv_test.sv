// tb_config_equiv_test.sv -- Verify config bus loading for fabric_add_tag.
//
// Loads a tag value (tag=5) into a fabric_add_tag module via the config
// bus, then sends untagged data through it and verifies the output tag
// matches the configured value.
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

module tb_config_equiv_test;

    // =========================================================================
    // Parameters
    // =========================================================================
    parameter DATA_WIDTH         = 32;
    parameter TAG_WIDTH          = 4;
    parameter EXPECTED_TAG       = 4'd5;
    parameter NUM_TOKENS         = 16;
    parameter CLK_PERIOD_NS      = 10;
    parameter RST_CYCLES         = 5;
    parameter SIM_TIMEOUT_CYCLES = 5000;

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
    // DUT: fabric_add_tag
    // =========================================================================
    logic                   cfg_valid;
    logic [31:0]            cfg_wdata;
    logic                   cfg_ready;

    logic                   in_valid;
    logic                   in_ready;
    logic [DATA_WIDTH-1:0]  in_data;

    logic                   out_valid;
    logic                   out_ready;
    logic [DATA_WIDTH-1:0]  out_data;
    logic [TAG_WIDTH-1:0]   out_tag;

    fabric_add_tag #(
        .DATA_WIDTH (DATA_WIDTH),
        .TAG_WIDTH  (TAG_WIDTH)
    ) u_dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .cfg_valid (cfg_valid),
        .cfg_wdata (cfg_wdata),
        .cfg_ready (cfg_ready),
        .in_valid  (in_valid),
        .in_ready  (in_ready),
        .in_data   (in_data),
        .out_valid (out_valid),
        .out_ready (out_ready),
        .out_data  (out_data),
        .out_tag   (out_tag)
    );

    // Output always ready (no backpressure for this test)
    assign out_ready = 1'b1;

    // =========================================================================
    // Config loading state machine: send one word with tag=EXPECTED_TAG
    // =========================================================================
    logic config_sent;

    always @(posedge clk or negedge rst_n) begin : cfg_driver_proc
        if (!rst_n) begin : cfg_driver_reset
            cfg_valid   <= 1'b0;
            cfg_wdata   <= 32'd0;
            config_sent <= 1'b0;
        end else begin : cfg_driver_active
            if (!config_sent) begin : cfg_send
                cfg_valid <= 1'b1;
                cfg_wdata <= {28'd0, EXPECTED_TAG};
                if (cfg_valid && cfg_ready) begin : cfg_accepted
                    config_sent <= 1'b1;
                    cfg_valid   <= 1'b0;
                end
            end else begin : cfg_idle
                cfg_valid <= 1'b0;
            end
        end
    end

    // =========================================================================
    // Data driver: send sequential values after config is loaded
    // =========================================================================
    integer send_idx;
    logic   send_done;

    always @(posedge clk or negedge rst_n) begin : data_driver_proc
        if (!rst_n) begin : data_driver_reset
            send_idx  <= 0;
            in_valid  <= 1'b0;
            in_data   <= '0;
            send_done <= 1'b0;
        end else begin : data_driver_active
            if (!config_sent || send_done) begin : data_wait
                in_valid <= 1'b0;
            end else begin : data_send
                in_data  <= send_idx[DATA_WIDTH-1:0];
                in_valid <= 1'b1;

                if (in_valid && in_ready) begin : data_advance
                    if (send_idx + 1 >= NUM_TOKENS) begin : data_last
                        send_done <= 1'b1;
                        in_valid  <= 1'b0;
                    end else begin : data_next
                        send_idx <= send_idx + 1;
                    end
                end
            end
        end
    end

    // =========================================================================
    // Output checker: verify data passthrough and tag == EXPECTED_TAG
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
                // Verify tag matches configured value
                if (out_tag !== EXPECTED_TAG) begin : checker_tag_mismatch
                    if (mismatch_count < 10) begin : checker_tag_report
                        $display("[tb_config_equiv_test] TAG MISMATCH[%0d]: got tag=%0d, expected tag=%0d",
                                 recv_idx, out_tag, EXPECTED_TAG);
                    end
                    mismatch_count <= mismatch_count + 1;
                end

                // Verify data passthrough
                if (out_data !== recv_idx[DATA_WIDTH-1:0]) begin : checker_data_mismatch
                    if (mismatch_count < 10) begin : checker_data_report
                        $display("[tb_config_equiv_test] DATA MISMATCH[%0d]: got %h, expected %h",
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
                    $display("[tb_config_equiv_test] PASS: All %0d tokens verified (tag=%0d, data passthrough correct)",
                             NUM_TOKENS, EXPECTED_TAG);
                end else begin : verdict_fail
                    $display("[tb_config_equiv_test] FAIL: %0d mismatches out of %0d tokens",
                             mismatch_count, NUM_TOKENS);
                end
                $finish;
            end

            if (cycle_count >= SIM_TIMEOUT_CYCLES) begin : timeout_hit
                $display("[tb_config_equiv_test] FAIL: Timeout at cycle %0d (received %0d/%0d tokens)",
                         cycle_count, recv_idx, NUM_TOKENS);
                $finish;
            end
        end
    end

endmodule
