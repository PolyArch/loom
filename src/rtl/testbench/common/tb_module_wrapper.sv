// tb_module_wrapper.sv -- Generic testbench wrapper template.
//
// Demonstrates how to wire together the shared testbench infrastructure
// modules for testing any DUT:
//   - tb_clk_rst_gen:     Clock and reset generation
//   - tb_config_loader:   Configuration loading into DUT
//   - tb_channel_driver:  Stimulus injection (one per DUT input port)
//   - tb_channel_monitor: Output capture (one per DUT output port)
//   - tb_backpressure_gen: Randomized backpressure on DUT outputs
//
// This wrapper includes:
//   - Golden trace comparison (pass/fail check)
//   - Simulation timeout
//   - Done detection (all drivers done + all expected outputs received)
//
// To adapt for a specific DUT:
//   1. Replace the DUT_PLACEHOLDER instantiation with the actual DUT
//   2. Adjust DATA_WIDTH, TAG_WIDTH, NUM_INPUTS, NUM_OUTPUTS parameters
//   3. Set trace file paths for each driver/monitor
//   4. Adjust GOLDEN_TOKENS to the expected output token count
//   5. Set SIM_TIMEOUT_CYCLES to an appropriate value
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

// Template module: many signals are intentionally unused in the placeholder
// configuration and will be connected when the DUT is wired in.
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNSIGNED */

module tb_module_wrapper;

    // =========================================================================
    // Parameters -- adjust per DUT
    // =========================================================================
    parameter DATA_WIDTH         = 32;
    parameter TAG_WIDTH          = 4;
    parameter CONFIG_WIDTH       = 32;
    parameter MAX_TOKENS         = 4096;
    parameter MAX_CONFIG_WORDS   = 1024;
    parameter CLK_PERIOD_NS      = 10;
    parameter RST_CYCLES         = 5;
    parameter SIM_TIMEOUT_CYCLES = 100000;

    // Number of input/output channels to the DUT (informational for adapters)
    parameter NUM_INPUTS  = 1;
    parameter NUM_OUTPUTS = 1;

    // Expected number of output tokens (for done detection)
    parameter GOLDEN_TOKENS = 0;

    // Backpressure configuration
    parameter BP_ENABLE        = 1;
    parameter BP_SEED          = 42;
    parameter BP_PROB_READY_PCT = 80;

    // Trace file paths (override via plusargs or per-test wrapper)
    parameter INPUT_TRACE_0  = "input_0.hex";
    parameter OUTPUT_TRACE_0 = "output_0.hex";
    parameter GOLDEN_TRACE_0 = "golden_0.hex";
    parameter CONFIG_FILE    = "config.hex";

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
    // Configuration loader
    // =========================================================================
    wire [CONFIG_WIDTH-1:0] cfg_wdata;
    wire                    cfg_valid;
    wire                    cfg_last;
    wire                    cfg_ready;
    wire                    config_done;

    tb_config_loader #(
        .MAX_CONFIG_WORDS (MAX_CONFIG_WORDS),
        .CONFIG_WIDTH     (CONFIG_WIDTH),
        .CONFIG_FILE      (CONFIG_FILE)
    ) u_config_loader (
        .clk         (clk),
        .rst_n       (rst_n),
        .cfg_wdata   (cfg_wdata),
        .cfg_valid   (cfg_valid),
        .cfg_last    (cfg_last),
        .cfg_ready   (cfg_ready),
        .config_done (config_done)
    );

    // =========================================================================
    // Input channel driver (channel 0)
    //
    // For multiple input channels, instantiate additional tb_channel_driver
    // modules with different trace files and connect to corresponding DUT
    // input ports.
    // =========================================================================
    wire [DATA_WIDTH-1:0] drv0_data;
    wire [TAG_WIDTH-1:0]  drv0_tag;
    wire                  drv0_valid;
    wire                  drv0_ready;
    wire                  drv0_done;
    wire [31:0]           drv0_token_count;

    tb_channel_driver #(
        .DATA_WIDTH (DATA_WIDTH),
        .TAG_WIDTH  (TAG_WIDTH),
        .MAX_TOKENS (MAX_TOKENS),
        .TRACE_FILE (INPUT_TRACE_0)
    ) u_driver_0 (
        .clk         (clk),
        .rst_n       (rst_n),
        .data        (drv0_data),
        .tag         (drv0_tag),
        .valid       (drv0_valid),
        .ready       (drv0_ready),
        .done        (drv0_done),
        .token_count (drv0_token_count)
    );

    // =========================================================================
    // Backpressure generator (for DUT output channel 0)
    //
    // Generates a pseudorandom ready signal to stress-test the DUT under
    // backpressure. When BP_ENABLE is 0, ready is always high.
    // =========================================================================
    wire bp0_ready;

    tb_backpressure_gen #(
        .SEED           (BP_SEED),
        .PROB_READY_PCT (BP_PROB_READY_PCT)
    ) u_bp_0 (
        .clk    (clk),
        .rst_n  (rst_n),
        .enable (BP_ENABLE[0]),
        .ready  (bp0_ready)
    );

    // =========================================================================
    // Output channel monitor (channel 0)
    //
    // Captures DUT output transfers for comparison against golden trace.
    // =========================================================================
    wire [DATA_WIDTH-1:0] mon0_data;
    wire [TAG_WIDTH-1:0]  mon0_tag;
    wire                  mon0_valid;
    wire                  mon0_ready;
    wire [31:0]           mon0_transfer_count;

    // Output ready driven by backpressure generator
    assign mon0_ready = bp0_ready;

    tb_channel_monitor #(
        .DATA_WIDTH (DATA_WIDTH),
        .TAG_WIDTH  (TAG_WIDTH),
        .MAX_TOKENS (MAX_TOKENS),
        .TRACE_FILE (OUTPUT_TRACE_0)
    ) u_monitor_0 (
        .clk            (clk),
        .rst_n          (rst_n),
        .data           (mon0_data),
        .tag            (mon0_tag),
        .valid          (mon0_valid),
        .ready          (mon0_ready),
        .transfer_count (mon0_transfer_count)
    );

    // =========================================================================
    // DUT instantiation placeholder
    //
    // Replace this section with the actual DUT module instance.
    // Wire the input/output channels and config bus to the DUT ports.
    //
    // Example for a single-input, single-output DUT:
    //
    //   my_dut #(
    //       .DATA_WIDTH (DATA_WIDTH),
    //       .TAG_WIDTH  (TAG_WIDTH)
    //   ) u_dut (
    //       .clk        (clk),
    //       .rst_n      (rst_n),
    //       // Config interface
    //       .cfg_wdata  (cfg_wdata),
    //       .cfg_valid  (cfg_valid),
    //       .cfg_last   (cfg_last),
    //       .cfg_ready  (cfg_ready),
    //       // Input channel
    //       .in_data    (drv0_data),
    //       .in_tag     (drv0_tag),
    //       .in_valid   (drv0_valid),
    //       .in_ready   (drv0_ready),
    //       // Output channel
    //       .out_data   (mon0_data),
    //       .out_tag    (mon0_tag),
    //       .out_valid  (mon0_valid),
    //       .out_ready  (mon0_ready)
    //   );
    //
    // =========================================================================

    // Stub wires for the placeholder (remove when DUT is connected)
    assign drv0_ready = 1'b1;
    assign cfg_ready  = 1'b1;
    assign mon0_data  = {DATA_WIDTH{1'b0}};
    assign mon0_tag   = {TAG_WIDTH{1'b0}};
    assign mon0_valid = 1'b0;

    // =========================================================================
    // Golden trace comparison
    // =========================================================================
    localparam ENTRY_WIDTH = DATA_WIDTH + TAG_WIDTH;
    reg [ENTRY_WIDTH-1:0] golden_mem [0:MAX_TOKENS-1];
    integer golden_count;
    integer mismatch_count;

    initial begin : golden_load
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < MAX_TOKENS; iter_var0 = iter_var0 + 1) begin : golden_init_loop
            golden_mem[iter_var0] = {ENTRY_WIDTH{1'b0}};
        end

        $readmemh(GOLDEN_TRACE_0, golden_mem);

        golden_count = 0;
        for (iter_var0 = 0; iter_var0 < MAX_TOKENS; iter_var0 = iter_var0 + 1) begin : golden_count_loop
            if (golden_mem[iter_var0] !== {ENTRY_WIDTH{1'bx}}) begin : golden_count_valid
                golden_count = iter_var0 + 1;
            end
        end

        $display("[tb_module_wrapper] Loaded %0d golden tokens from %s",
                 golden_count, GOLDEN_TRACE_0);
    end

    // =========================================================================
    // Simulation timeout and done detection
    // =========================================================================
    integer cycle_count;
    reg     sim_done;
    reg     sim_pass;

    always @(posedge clk or negedge rst_n) begin : timeout_proc
        if (!rst_n) begin : timeout_reset
            cycle_count <= 0;
            sim_done    <= 1'b0;
        end else begin : timeout_active
            cycle_count <= cycle_count + 1;

            // Done: all drivers finished and enough output tokens collected
            if (!sim_done && config_done && drv0_done) begin : check_done
                if (golden_count > 0) begin : check_golden_done
                    if (mon0_transfer_count >= golden_count[31:0]) begin : all_outputs_received
                        sim_done <= 1'b1;
                    end
                end else if (GOLDEN_TOKENS > 0) begin : check_param_done
                    if (mon0_transfer_count >= GOLDEN_TOKENS[31:0]) begin : param_outputs_received
                        sim_done <= 1'b1;
                    end
                end else begin : check_drv_done
                    // No golden count specified; finish when driver is done
                    sim_done <= 1'b1;
                end
            end

            // Timeout guard
            if (cycle_count >= SIM_TIMEOUT_CYCLES) begin : timeout_hit
                $display("[tb_module_wrapper] ERROR: Simulation timeout at cycle %0d", cycle_count);
                sim_done <= 1'b1;
            end
        end
    end

    // =========================================================================
    // Final comparison and verdict
    // =========================================================================
    always @(posedge clk) begin : verdict_proc
        if (sim_done) begin : verdict_check
            compare_and_finish;
        end
    end

    task compare_and_finish;
        begin : compare_body
            integer iter_var0;
            mismatch_count = 0;

            $display("[tb_module_wrapper] --- Comparison ---");
            $display("[tb_module_wrapper] Output transfers: %0d, Golden tokens: %0d",
                     mon0_transfer_count, golden_count);

            if (golden_count > 0) begin : compare_with_golden
                // Check token count match
                if (mon0_transfer_count != golden_count[31:0]) begin : count_mismatch
                    $display("[tb_module_wrapper] FAIL: Token count mismatch (got %0d, expected %0d)",
                             mon0_transfer_count, golden_count);
                    mismatch_count = mismatch_count + 1;
                end

                // Compare captured tokens against golden
                for (iter_var0 = 0; iter_var0 < golden_count && iter_var0 < MAX_TOKENS;
                     iter_var0 = iter_var0 + 1) begin : compare_loop
                    if (u_monitor_0.capture_mem[iter_var0] !== golden_mem[iter_var0]) begin : token_mismatch
                        if (mismatch_count < 10) begin : report_mismatch
                            $display("[tb_module_wrapper] MISMATCH at token %0d: got %h, expected %h",
                                     iter_var0,
                                     u_monitor_0.capture_mem[iter_var0],
                                     golden_mem[iter_var0]);
                        end
                        mismatch_count = mismatch_count + 1;
                    end
                end

                if (mismatch_count == 0) begin : pass_verdict
                    $display("[tb_module_wrapper] PASS: All %0d tokens match", golden_count);
                    sim_pass = 1'b1;
                end else begin : fail_verdict
                    $display("[tb_module_wrapper] FAIL: %0d mismatches found", mismatch_count);
                    sim_pass = 1'b0;
                end
            end else begin : no_golden
                $display("[tb_module_wrapper] WARN: No golden trace loaded; skipping comparison");
                sim_pass = 1'b1;
            end

            $display("[tb_module_wrapper] Simulation ended at cycle %0d", cycle_count);

            if (sim_pass) begin : exit_pass
                $finish(0);
            end else begin : exit_fail
                $finish(1);
            end
        end
    endtask

/* verilator lint_on UNUSEDSIGNAL */
/* verilator lint_on UNUSEDPARAM */
/* verilator lint_on UNSIGNED */

endmodule
