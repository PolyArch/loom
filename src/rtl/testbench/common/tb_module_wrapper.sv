// tb_module_wrapper.sv -- Generic testbench wrapper template.
//
// Wires together shared testbench infrastructure for testing any DUT:
//   - tb_clk_rst_gen, tb_config_loader, tb_channel_driver,
//     tb_channel_monitor, tb_backpressure_gen
//
// Trace format: all traces use packed hex (one value per line).
//   - Input trace: {tag, data} per line (or just data if TAG_WIDTH==0)
//   - Output trace: same format
//   - Golden trace: same format
//   - Config trace: one 32-bit word per line
//
// Token counts (NUM_INPUT_TOKENS, GOLDEN_TOKENS, NUM_CONFIG_WORDS) are
// provided explicitly as parameters -- no auto-detection from files.
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNSIGNED */

module tb_module_wrapper;

    // =========================================================================
    // Parameters -- adjust per DUT
    // =========================================================================
    parameter DATA_WIDTH         = 32;
    parameter TAG_WIDTH          = 0;
    parameter CONFIG_WIDTH       = 32;
    parameter MAX_TOKENS         = 4096;
    parameter MAX_CONFIG_WORDS   = 1024;
    parameter CLK_PERIOD_NS      = 10;
    parameter RST_CYCLES         = 5;
    parameter SIM_TIMEOUT_CYCLES = 100000;

    // Explicit token/word counts (no auto-detection)
    parameter NUM_INPUT_TOKENS  = 0;
    parameter GOLDEN_TOKENS     = 0;
    parameter NUM_CONFIG_WORDS  = 0;

    // Backpressure configuration
    parameter BP_ENABLE         = 1;
    parameter BP_SEED           = 42;
    parameter BP_PROB_READY_PCT = 80;

    // Trace file paths
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
        .NUM_WORDS        (NUM_CONFIG_WORDS),
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
    // =========================================================================
    localparam TAG_W = (TAG_WIDTH > 0) ? TAG_WIDTH : 1;

    wire [DATA_WIDTH-1:0] drv0_data;
    wire [TAG_W-1:0]      drv0_tag;
    wire                  drv0_valid;
    wire                  drv0_ready;
    wire                  drv0_done;
    wire [31:0]           drv0_token_count;

    tb_channel_driver #(
        .DATA_WIDTH (DATA_WIDTH),
        .TAG_WIDTH  (TAG_WIDTH),
        .NUM_TOKENS (NUM_INPUT_TOKENS),
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
    // Backpressure generator
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
    // =========================================================================
    wire [DATA_WIDTH-1:0] mon0_data;
    wire [TAG_W-1:0]      mon0_tag;
    wire                  mon0_valid;
    wire                  mon0_ready;
    wire [31:0]           mon0_transfer_count;

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
    // Wire input/output channels and config bus to DUT ports.
    // =========================================================================

    assign drv0_ready = 1'b1;
    assign cfg_ready  = 1'b1;
    assign mon0_data  = '0;
    assign mon0_tag   = '0;
    assign mon0_valid = 1'b0;

    // =========================================================================
    // Golden trace comparison
    // =========================================================================
    localparam ENTRY_WIDTH = DATA_WIDTH + TAG_W;
    reg [ENTRY_WIDTH-1:0] golden_mem [0:MAX_TOKENS-1];
    integer mismatch_count;

    initial begin : golden_load
        if (GOLDEN_TOKENS > 0) begin : do_load_golden
            $readmemh(GOLDEN_TRACE_0, golden_mem, 0, GOLDEN_TOKENS - 1);
            $display("[tb_module_wrapper] Loaded %0d golden tokens from %s",
                     GOLDEN_TOKENS, GOLDEN_TRACE_0);
        end else begin : no_golden_load
            $display("[tb_module_wrapper] No golden tokens specified");
        end
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

            if (!sim_done && config_done && drv0_done) begin : check_done
                if (GOLDEN_TOKENS > 0) begin : check_golden_done
                    if (mon0_transfer_count >= GOLDEN_TOKENS[31:0]) begin : all_out
                        sim_done <= 1'b1;
                    end
                end else begin : check_drv_done
                    sim_done <= 1'b1;
                end
            end

            if (cycle_count >= SIM_TIMEOUT_CYCLES) begin : timeout_hit
                $display("[tb_module_wrapper] ERROR: Timeout at cycle %0d", cycle_count);
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

            $display("[tb_module_wrapper] Output: %0d transfers, Golden: %0d tokens",
                     mon0_transfer_count, GOLDEN_TOKENS);

            if (GOLDEN_TOKENS > 0) begin : compare_with_golden
                if (mon0_transfer_count != GOLDEN_TOKENS[31:0]) begin : count_mismatch
                    $display("[tb_module_wrapper] FAIL: count mismatch (got %0d, expected %0d)",
                             mon0_transfer_count, GOLDEN_TOKENS);
                    mismatch_count = mismatch_count + 1;
                end

                for (iter_var0 = 0; iter_var0 < GOLDEN_TOKENS && iter_var0 < MAX_TOKENS;
                     iter_var0 = iter_var0 + 1) begin : compare_loop
                    if (u_monitor_0.capture_mem[iter_var0] !== golden_mem[iter_var0]) begin : token_mismatch
                        if (mismatch_count < 10) begin : report_mismatch
                            $display("[tb_module_wrapper] MISMATCH[%0d]: got %h, expected %h",
                                     iter_var0,
                                     u_monitor_0.capture_mem[iter_var0],
                                     golden_mem[iter_var0]);
                        end
                        mismatch_count = mismatch_count + 1;
                    end
                end

                if (mismatch_count == 0) begin : pass_verdict
                    $display("[tb_module_wrapper] PASS: All %0d tokens match", GOLDEN_TOKENS);
                    sim_pass = 1'b1;
                end else begin : fail_verdict
                    $display("[tb_module_wrapper] FAIL: %0d mismatches", mismatch_count);
                    sim_pass = 1'b0;
                end
            end else begin : no_golden
                $display("[tb_module_wrapper] WARN: No golden trace; skipping comparison");
                sim_pass = 1'b1;
            end

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
