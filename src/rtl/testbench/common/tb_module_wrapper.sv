// tb_module_wrapper.sv -- Generic testbench wrapper template.
//
// Wires together shared testbench infrastructure for testing any DUT:
//   - tb_clk_rst_gen, tb_config_loader, tb_channel_driver,
//     tb_channel_monitor, tb_backpressure_gen
//
// Supports up to MAX_CHANNELS (4) input channels and MAX_CHANNELS output
// channels via generate loops. Each channel reads its own plusarg for
// trace file path and token count.
//
// The DUT is instantiated via a generated include file (dut_inst.svh)
// produced by run_rtl_checks.py. This file wires the DUT module ports
// to the testbench driver/monitor arrays. When DUT_MODULE is defined
// but no dut_inst.svh is provided, the wrapper uses a 1in/1out default.
// When DUT_MODULE is not defined, a loopback stub is used.
//
// Trace format: all traces use packed hex (one value per line).
//   - Input trace: {tag, data} per line (or just data if TAG_WIDTH==0)
//   - Output trace: same format
//   - Golden trace: same format
//   - Config trace: one 32-bit word per line
//
// Runtime-configurable via Verilator/VCS plusargs:
//   +INPUT_TRACE_<i>=path   (per input channel)
//   +OUTPUT_TRACE_<i>=path  (per output channel)
//   +GOLDEN_TRACE_<i>=path  (per output channel)
//   +NUM_INPUT_TOKENS_<i>=N (per input channel)
//   +GOLDEN_TOKENS_<i>=N    (per output channel)
//   +CONFIG_FILE=path
//   +NUM_CONFIG_WORDS=N
//   +SIM_TIMEOUT_CYCLES=N
//
// Token counts are provided explicitly as plusargs -- no auto-detection
// from files.
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNSIGNED */

module tb_module_wrapper;

    // =========================================================================
    // Parameters -- adjust per DUT (compile-time defaults)
    // =========================================================================
    parameter DATA_WIDTH         = 32;
    parameter TAG_WIDTH          = 0;
    parameter CONFIG_WIDTH       = 32;
    parameter MAX_TOKENS         = 4096;
    parameter MAX_CONFIG_WORDS   = 1024;
    parameter CLK_PERIOD_NS      = 10;
    parameter RST_CYCLES         = 5;
    parameter SIM_TIMEOUT_CYCLES = 100000;

    // Multi-channel support: up to 4 inputs and 4 outputs.
    // These are set at compile time via +define+NUM_DUT_INPUTS=N etc.
`ifdef NUM_DUT_INPUTS
    parameter NUM_DUT_INPUTS     = `NUM_DUT_INPUTS;
`else
    parameter NUM_DUT_INPUTS     = 1;
`endif
`ifdef NUM_DUT_OUTPUTS
    parameter NUM_DUT_OUTPUTS    = `NUM_DUT_OUTPUTS;
`else
    parameter NUM_DUT_OUTPUTS    = 1;
`endif

    // Explicit token/word counts (compile-time defaults, per-channel
    // overrides come from plusargs)
    parameter NUM_INPUT_TOKENS  = 0;
    parameter GOLDEN_TOKENS     = 0;
    parameter NUM_CONFIG_WORDS  = 0;

    // Backpressure configuration
    parameter BP_ENABLE         = 1;
    parameter BP_SEED           = 42;
    parameter BP_PROB_READY_PCT = 80;

    // Trace file paths (compile-time defaults, overridable via plusargs)
    parameter INPUT_TRACE_0  = "input_0.hex";
    parameter OUTPUT_TRACE_0 = "output_0.hex";
    parameter GOLDEN_TRACE_0 = "golden_0.hex";
    parameter CONFIG_FILE    = "config.hex";

    // =========================================================================
    // Constants
    // =========================================================================
    localparam TAG_W = (TAG_WIDTH > 0) ? TAG_WIDTH : 1;
    localparam MAX_CHANNELS = 4;
    localparam ENTRY_WIDTH = DATA_WIDTH + TAG_W;

    // =========================================================================
    // Plusarg overrides -- runtime configuration via +NAME=value
    //
    // Per-channel trace paths and token counts.
    // =========================================================================
    reg [4096*8-1:0] plusarg_input_trace_0;
    reg [4096*8-1:0] plusarg_input_trace_1;
    reg [4096*8-1:0] plusarg_input_trace_2;
    reg [4096*8-1:0] plusarg_input_trace_3;
    reg [4096*8-1:0] plusarg_output_trace_0;
    reg [4096*8-1:0] plusarg_output_trace_1;
    reg [4096*8-1:0] plusarg_output_trace_2;
    reg [4096*8-1:0] plusarg_output_trace_3;
    reg [4096*8-1:0] plusarg_golden_trace_0;
    reg [4096*8-1:0] plusarg_golden_trace_1;
    reg [4096*8-1:0] plusarg_golden_trace_2;
    reg [4096*8-1:0] plusarg_golden_trace_3;
    reg [4096*8-1:0] plusarg_config_file;
    integer          plusarg_num_input_tokens_0;
    integer          plusarg_num_input_tokens_1;
    integer          plusarg_num_input_tokens_2;
    integer          plusarg_num_input_tokens_3;
    integer          plusarg_golden_tokens_0;
    integer          plusarg_golden_tokens_1;
    integer          plusarg_golden_tokens_2;
    integer          plusarg_golden_tokens_3;
    integer          plusarg_num_config_words;
    integer          plusarg_sim_timeout;

    // Effective values after plusarg resolution
    reg [4096*8-1:0] eff_input_trace  [0:3];
    reg [4096*8-1:0] eff_output_trace [0:3];
    reg [4096*8-1:0] eff_golden_trace [0:3];
    reg [4096*8-1:0] eff_config_file;
    integer          eff_num_input_tokens [0:3];
    integer          eff_golden_tokens    [0:3];
    integer          eff_num_config_words;
    integer          eff_sim_timeout;

    initial begin : plusarg_init
        // Defaults for all channels
        eff_input_trace[0]  = INPUT_TRACE_0;
        eff_input_trace[1]  = "input_1.hex";
        eff_input_trace[2]  = "input_2.hex";
        eff_input_trace[3]  = "input_3.hex";
        eff_output_trace[0] = OUTPUT_TRACE_0;
        eff_output_trace[1] = "output_1.hex";
        eff_output_trace[2] = "output_2.hex";
        eff_output_trace[3] = "output_3.hex";
        eff_golden_trace[0] = GOLDEN_TRACE_0;
        eff_golden_trace[1] = "golden_1.hex";
        eff_golden_trace[2] = "golden_2.hex";
        eff_golden_trace[3] = "golden_3.hex";

        // Default per-channel token counts from compile-time parameters.
        // The single NUM_INPUT_TOKENS / GOLDEN_TOKENS params are used as
        // defaults for all channels; per-channel plusargs override them.
        eff_num_input_tokens[0] = NUM_INPUT_TOKENS;
        eff_num_input_tokens[1] = NUM_INPUT_TOKENS;
        eff_num_input_tokens[2] = NUM_INPUT_TOKENS;
        eff_num_input_tokens[3] = NUM_INPUT_TOKENS;
        eff_golden_tokens[0]    = GOLDEN_TOKENS;
        eff_golden_tokens[1]    = GOLDEN_TOKENS;
        eff_golden_tokens[2]    = GOLDEN_TOKENS;
        eff_golden_tokens[3]    = GOLDEN_TOKENS;

        // Input trace plusargs
        if ($value$plusargs("INPUT_TRACE_0=%s", plusarg_input_trace_0)) begin : use_in0
            eff_input_trace[0] = plusarg_input_trace_0;
        end
        if ($value$plusargs("INPUT_TRACE_1=%s", plusarg_input_trace_1)) begin : use_in1
            eff_input_trace[1] = plusarg_input_trace_1;
        end
        if ($value$plusargs("INPUT_TRACE_2=%s", plusarg_input_trace_2)) begin : use_in2
            eff_input_trace[2] = plusarg_input_trace_2;
        end
        if ($value$plusargs("INPUT_TRACE_3=%s", plusarg_input_trace_3)) begin : use_in3
            eff_input_trace[3] = plusarg_input_trace_3;
        end

        // Output trace plusargs
        if ($value$plusargs("OUTPUT_TRACE_0=%s", plusarg_output_trace_0)) begin : use_out0
            eff_output_trace[0] = plusarg_output_trace_0;
        end
        if ($value$plusargs("OUTPUT_TRACE_1=%s", plusarg_output_trace_1)) begin : use_out1
            eff_output_trace[1] = plusarg_output_trace_1;
        end
        if ($value$plusargs("OUTPUT_TRACE_2=%s", plusarg_output_trace_2)) begin : use_out2
            eff_output_trace[2] = plusarg_output_trace_2;
        end
        if ($value$plusargs("OUTPUT_TRACE_3=%s", plusarg_output_trace_3)) begin : use_out3
            eff_output_trace[3] = plusarg_output_trace_3;
        end

        // Golden trace plusargs
        if ($value$plusargs("GOLDEN_TRACE_0=%s", plusarg_golden_trace_0)) begin : use_gld0
            eff_golden_trace[0] = plusarg_golden_trace_0;
        end
        if ($value$plusargs("GOLDEN_TRACE_1=%s", plusarg_golden_trace_1)) begin : use_gld1
            eff_golden_trace[1] = plusarg_golden_trace_1;
        end
        if ($value$plusargs("GOLDEN_TRACE_2=%s", plusarg_golden_trace_2)) begin : use_gld2
            eff_golden_trace[2] = plusarg_golden_trace_2;
        end
        if ($value$plusargs("GOLDEN_TRACE_3=%s", plusarg_golden_trace_3)) begin : use_gld3
            eff_golden_trace[3] = plusarg_golden_trace_3;
        end

        // Per-channel input token count plusargs
        if ($value$plusargs("NUM_INPUT_TOKENS_0=%d", plusarg_num_input_tokens_0)) begin : use_in_tok0
            eff_num_input_tokens[0] = plusarg_num_input_tokens_0;
        end
        if ($value$plusargs("NUM_INPUT_TOKENS_1=%d", plusarg_num_input_tokens_1)) begin : use_in_tok1
            eff_num_input_tokens[1] = plusarg_num_input_tokens_1;
        end
        if ($value$plusargs("NUM_INPUT_TOKENS_2=%d", plusarg_num_input_tokens_2)) begin : use_in_tok2
            eff_num_input_tokens[2] = plusarg_num_input_tokens_2;
        end
        if ($value$plusargs("NUM_INPUT_TOKENS_3=%d", plusarg_num_input_tokens_3)) begin : use_in_tok3
            eff_num_input_tokens[3] = plusarg_num_input_tokens_3;
        end

        // Per-channel golden token count plusargs
        if ($value$plusargs("GOLDEN_TOKENS_0=%d", plusarg_golden_tokens_0)) begin : use_gld_tok0
            eff_golden_tokens[0] = plusarg_golden_tokens_0;
        end
        if ($value$plusargs("GOLDEN_TOKENS_1=%d", plusarg_golden_tokens_1)) begin : use_gld_tok1
            eff_golden_tokens[1] = plusarg_golden_tokens_1;
        end
        if ($value$plusargs("GOLDEN_TOKENS_2=%d", plusarg_golden_tokens_2)) begin : use_gld_tok2
            eff_golden_tokens[2] = plusarg_golden_tokens_2;
        end
        if ($value$plusargs("GOLDEN_TOKENS_3=%d", plusarg_golden_tokens_3)) begin : use_gld_tok3
            eff_golden_tokens[3] = plusarg_golden_tokens_3;
        end

        // Config file plusarg
        eff_config_file = CONFIG_FILE;
        if ($value$plusargs("CONFIG_FILE=%s", plusarg_config_file)) begin : use_config_file
            eff_config_file = plusarg_config_file;
        end

        // Config word count
        eff_num_config_words = NUM_CONFIG_WORDS;
        if ($value$plusargs("NUM_CONFIG_WORDS=%d", plusarg_num_config_words)) begin : use_config_words
            eff_num_config_words = plusarg_num_config_words;
        end

        // Simulation timeout
        eff_sim_timeout = SIM_TIMEOUT_CYCLES;
        if ($value$plusargs("SIM_TIMEOUT_CYCLES=%d", plusarg_sim_timeout)) begin : use_sim_timeout
            eff_sim_timeout = plusarg_sim_timeout;
        end
    end

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
    // Multi-channel wires -- sized for up to MAX_CHANNELS channels
    // =========================================================================

    // Input channel wires (from drivers to DUT)
    wire [DATA_WIDTH-1:0] drv_data  [0:MAX_CHANNELS-1];
    wire [TAG_W-1:0]      drv_tag   [0:MAX_CHANNELS-1];
    wire                  drv_valid [0:MAX_CHANNELS-1];
    wire                  drv_ready [0:MAX_CHANNELS-1];
    wire                  drv_done  [0:MAX_CHANNELS-1];
    wire [31:0]           drv_token_count [0:MAX_CHANNELS-1];

    // Output channel wires (from DUT to monitors)
    wire [DATA_WIDTH-1:0] mon_data  [0:MAX_CHANNELS-1];
    wire [TAG_W-1:0]      mon_tag   [0:MAX_CHANNELS-1];
    wire                  mon_valid [0:MAX_CHANNELS-1];
    wire                  mon_ready [0:MAX_CHANNELS-1];
    wire [31:0]           mon_transfer_count [0:MAX_CHANNELS-1];

    // Backpressure ready signals
    wire                  bp_ready  [0:MAX_CHANNELS-1];

    // =========================================================================
    // Generate input channel drivers (one per NUM_DUT_INPUTS)
    //
    // Each driver uses per-channel plusargs for trace file and token count.
    // =========================================================================
    generate
        genvar gi;
        for (gi = 0; gi < MAX_CHANNELS; gi = gi + 1) begin : gen_input_ch
            if (gi < NUM_DUT_INPUTS) begin : active_input
                tb_channel_driver #(
                    .DATA_WIDTH (DATA_WIDTH),
                    .TAG_WIDTH  (TAG_WIDTH),
                    .NUM_TOKENS (NUM_INPUT_TOKENS),
                    .MAX_TOKENS (MAX_TOKENS),
                    .TRACE_FILE (INPUT_TRACE_0)
                ) u_driver (
                    .clk         (clk),
                    .rst_n       (rst_n),
                    .data        (drv_data[gi]),
                    .tag         (drv_tag[gi]),
                    .valid       (drv_valid[gi]),
                    .ready       (drv_ready[gi]),
                    .done        (drv_done[gi]),
                    .token_count (drv_token_count[gi])
                );

                // Override the driver's trace file and token count at runtime
                // using the per-channel plusarg-resolved values.
                initial begin : override_driver
                    #0; // ensure plusarg_init has executed
                    u_driver.eff_num_tokens = eff_num_input_tokens[gi];
                    if (eff_num_input_tokens[gi] > 0) begin : do_reload
                        $readmemh(eff_input_trace[gi], u_driver.token_mem,
                                  0, eff_num_input_tokens[gi] - 1);
                        $display("[tb_module_wrapper] Driver[%0d]: %0d tokens from %s",
                                 gi, eff_num_input_tokens[gi], eff_input_trace[gi]);
                    end
                end
            end else begin : stub_input
                assign drv_data[gi]        = '0;
                assign drv_tag[gi]         = '0;
                assign drv_valid[gi]       = 1'b0;
                assign drv_done[gi]        = 1'b1;
                assign drv_token_count[gi] = 32'd0;
            end
        end
    endgenerate

    // =========================================================================
    // Generate backpressure generators (one per output channel)
    // =========================================================================
    generate
        genvar gb;
        for (gb = 0; gb < MAX_CHANNELS; gb = gb + 1) begin : gen_bp_ch
            if (gb < NUM_DUT_OUTPUTS) begin : active_bp
                tb_backpressure_gen #(
                    .SEED           (BP_SEED + gb),
                    .PROB_READY_PCT (BP_PROB_READY_PCT)
                ) u_bp (
                    .clk    (clk),
                    .rst_n  (rst_n),
                    .enable (BP_ENABLE[0]),
                    .ready  (bp_ready[gb])
                );
            end else begin : stub_bp
                assign bp_ready[gb] = 1'b1;
            end
        end
    endgenerate

    // =========================================================================
    // Generate output channel monitors (one per NUM_DUT_OUTPUTS)
    // =========================================================================
    generate
        genvar go;
        for (go = 0; go < MAX_CHANNELS; go = go + 1) begin : gen_output_ch
            if (go < NUM_DUT_OUTPUTS) begin : active_output
                // Connect backpressure to monitor ready
                assign mon_ready[go] = bp_ready[go];

                tb_channel_monitor #(
                    .DATA_WIDTH (DATA_WIDTH),
                    .TAG_WIDTH  (TAG_WIDTH),
                    .MAX_TOKENS (MAX_TOKENS),
                    .TRACE_FILE (OUTPUT_TRACE_0)
                ) u_monitor (
                    .clk            (clk),
                    .rst_n          (rst_n),
                    .data           (mon_data[go]),
                    .tag            (mon_tag[go]),
                    .valid          (mon_valid[go]),
                    .ready          (mon_ready[go]),
                    .transfer_count (mon_transfer_count[go])
                );

                // Override the monitor's output file at runtime
                initial begin : override_monitor
                    #0; // ensure plusarg_init has executed
                    // Close the default file and reopen with the plusarg path
                    if (u_monitor.file_open) begin : close_default
                        $fclose(u_monitor.fd);
                    end
                    u_monitor.fd = $fopen(eff_output_trace[go], "w");
                    if (u_monitor.fd == 0) begin : reopen_fail
                        $display("[tb_module_wrapper] ERROR: Monitor[%0d] cannot open %s",
                                 go, eff_output_trace[go]);
                        u_monitor.file_open = 1'b0;
                    end else begin : reopen_ok
                        u_monitor.file_open = 1'b1;
                        $display("[tb_module_wrapper] Monitor[%0d]: writing to %s",
                                 go, eff_output_trace[go]);
                    end
                end
            end else begin : stub_output
                assign mon_ready[go]          = 1'b1;
                assign mon_transfer_count[go] = 32'd0;
            end
        end
    endgenerate

    // =========================================================================
    // Configuration loader runtime override
    // =========================================================================
    initial begin : override_config_loader
        #0; // ensure plusarg_init has executed
        u_config_loader.eff_num_words = eff_num_config_words;
        if (eff_num_config_words > 0) begin : do_reload_cfg
            $readmemh(eff_config_file, u_config_loader.config_mem,
                      0, eff_num_config_words - 1);
            $display("[tb_module_wrapper] Config: %0d words from %s",
                     eff_num_config_words, eff_config_file);
        end
    end

    // =========================================================================
    // DUT instantiation
    //
    // The DUT is instantiated via a generated include file (dut_inst.svh).
    // This file is produced per-test by the Python runner (run_rtl_checks.py)
    // and contains the DUT module instantiation with the correct port
    // connections for the specific port topology.
    //
    // The include file connects:
    //   clk, rst_n
    //   drv_data[i], drv_valid[i], drv_ready[i]  -- for each input port
    //   mon_data[j], mon_valid[j], mon_ready[j]  -- for each output port
    //   cfg_valid, cfg_wdata, cfg_last, cfg_ready -- config bus
    //
    // It also ties off drv_ready[i] for unused input channels.
    //
    // If DUT_MODULE is defined but no dut_inst.svh is found, a default
    // 1-input 1-output instantiation is used.
    // =========================================================================

`ifdef DUT_MODULE
`ifdef DUT_INST_SVH
    `include "dut_inst.svh"
`else
    // Default: 1-input, 1-output (backward compatible)
    `DUT_MODULE u_dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .mod_in0        (drv_data[0]),
        .mod_in0_valid  (drv_valid[0]),
        .mod_in0_ready  (drv_ready[0]),
        .mod_out0       (mon_data[0]),
        .mod_out0_valid (mon_valid[0]),
        .mod_out0_ready (mon_ready[0]),
        .cfg_valid      (cfg_valid),
        .cfg_wdata      (cfg_wdata),
        .cfg_last       (cfg_last),
        .cfg_ready      (cfg_ready)
    );

    // Tie off unused input channel ready signals
    generate
        genvar gtr;
        for (gtr = 0; gtr < MAX_CHANNELS; gtr = gtr + 1) begin : gen_tieoff_ready
            if (gtr >= NUM_DUT_INPUTS) begin : tieoff_drv_ready
                assign drv_ready[gtr] = 1'b1;
            end
        end
    endgenerate
`endif

`else
    // Fallback loopback stub for TB infrastructure self-test.
    // Input is consumed immediately; output side stays idle.
    generate
        genvar gstub;
        for (gstub = 0; gstub < MAX_CHANNELS; gstub = gstub + 1) begin : gen_stub
            assign drv_ready[gstub] = 1'b1;
            assign mon_data[gstub]  = '0;
            assign mon_tag[gstub]   = '0;
            assign mon_valid[gstub] = 1'b0;
        end
    endgenerate
    assign cfg_ready = 1'b1;
`endif

    // =========================================================================
    // Per-channel golden trace comparison
    //
    // Each output channel has its own golden memory loaded from per-channel
    // plusargs. The comparison checks ALL output channels, not just channel 0.
    // =========================================================================
    reg [ENTRY_WIDTH-1:0] golden_mem_0 [0:MAX_TOKENS-1];
    reg [ENTRY_WIDTH-1:0] golden_mem_1 [0:MAX_TOKENS-1];
    reg [ENTRY_WIDTH-1:0] golden_mem_2 [0:MAX_TOKENS-1];
    reg [ENTRY_WIDTH-1:0] golden_mem_3 [0:MAX_TOKENS-1];

    initial begin : golden_load
        #0; // ensure plusarg_init has executed
        if (eff_golden_tokens[0] > 0) begin : do_load_golden_0
            $readmemh(eff_golden_trace[0], golden_mem_0, 0,
                      eff_golden_tokens[0] - 1);
            $display("[tb_module_wrapper] Loaded %0d golden tokens ch0 from %s",
                     eff_golden_tokens[0], eff_golden_trace[0]);
        end
        if (eff_golden_tokens[1] > 0) begin : do_load_golden_1
            $readmemh(eff_golden_trace[1], golden_mem_1, 0,
                      eff_golden_tokens[1] - 1);
            $display("[tb_module_wrapper] Loaded %0d golden tokens ch1 from %s",
                     eff_golden_tokens[1], eff_golden_trace[1]);
        end
        if (eff_golden_tokens[2] > 0) begin : do_load_golden_2
            $readmemh(eff_golden_trace[2], golden_mem_2, 0,
                      eff_golden_tokens[2] - 1);
            $display("[tb_module_wrapper] Loaded %0d golden tokens ch2 from %s",
                     eff_golden_tokens[2], eff_golden_trace[2]);
        end
        if (eff_golden_tokens[3] > 0) begin : do_load_golden_3
            $readmemh(eff_golden_trace[3], golden_mem_3, 0,
                      eff_golden_tokens[3] - 1);
            $display("[tb_module_wrapper] Loaded %0d golden tokens ch3 from %s",
                     eff_golden_tokens[3], eff_golden_trace[3]);
        end
    end

    // =========================================================================
    // Simulation timeout and done detection
    //
    // Done when all input drivers have finished AND all output channels
    // that have golden tokens have received the expected count.
    // =========================================================================
    integer cycle_count;
    reg     sim_done;
    reg     sim_pass;

    // All input drivers done
    wire all_drivers_done;
    assign all_drivers_done = drv_done[0] & drv_done[1] & drv_done[2] & drv_done[3];

    // Per-channel output done: each output channel with golden tokens
    // must have received at least that many transfers.
    wire out_ch0_done;
    wire out_ch1_done;
    wire out_ch2_done;
    wire out_ch3_done;

    assign out_ch0_done = (eff_golden_tokens[0] <= 0) ||
                          (mon_transfer_count[0] >= eff_golden_tokens[0][31:0]);
    assign out_ch1_done = (eff_golden_tokens[1] <= 0) ||
                          (mon_transfer_count[1] >= eff_golden_tokens[1][31:0]);
    assign out_ch2_done = (eff_golden_tokens[2] <= 0) ||
                          (mon_transfer_count[2] >= eff_golden_tokens[2][31:0]);
    assign out_ch3_done = (eff_golden_tokens[3] <= 0) ||
                          (mon_transfer_count[3] >= eff_golden_tokens[3][31:0]);

    wire all_outputs_done;
    assign all_outputs_done = out_ch0_done & out_ch1_done &
                              out_ch2_done & out_ch3_done;

    // Track whether any channel has golden tokens (for done gating)
    wire has_any_golden;
    assign has_any_golden = (eff_golden_tokens[0] > 0) ||
                            (eff_golden_tokens[1] > 0) ||
                            (eff_golden_tokens[2] > 0) ||
                            (eff_golden_tokens[3] > 0);

    always @(posedge clk or negedge rst_n) begin : timeout_proc
        if (!rst_n) begin : timeout_reset
            cycle_count <= 0;
            sim_done    <= 1'b0;
        end else begin : timeout_active
            cycle_count <= cycle_count + 1;

            if (!sim_done && config_done && all_drivers_done) begin : check_done
                if (has_any_golden) begin : check_golden_done
                    if (all_outputs_done) begin : all_out
                        sim_done <= 1'b1;
                    end
                end else begin : check_drv_done
                    sim_done <= 1'b1;
                end
            end

            if (cycle_count >= eff_sim_timeout) begin : timeout_hit
                $display("[tb_module_wrapper] ERROR: Timeout at cycle %0d", cycle_count);
                sim_done <= 1'b1;
            end
        end
    end

    // =========================================================================
    // Final comparison and verdict (compares ALL output channels)
    // =========================================================================
    always @(posedge clk) begin : verdict_proc
        if (sim_done) begin : verdict_check
            compare_and_finish;
        end
    end

    task compare_and_finish;
        begin : compare_body
            integer iter_var0;
            integer iter_var1;
            integer ch_mismatch;
            integer total_mismatch;
            integer total_golden;

            total_mismatch = 0;
            total_golden = 0;

            // Compare each output channel that has golden tokens
            for (iter_var1 = 0; iter_var1 < NUM_DUT_OUTPUTS && iter_var1 < MAX_CHANNELS;
                 iter_var1 = iter_var1 + 1) begin : compare_channels
                if (eff_golden_tokens[iter_var1] > 0) begin : compare_ch
                    ch_mismatch = 0;
                    total_golden = total_golden + eff_golden_tokens[iter_var1];

                    $display("[tb_module_wrapper] Output[%0d]: %0d transfers, Golden: %0d tokens",
                             iter_var1, mon_transfer_count[iter_var1],
                             eff_golden_tokens[iter_var1]);

                    // Check transfer count
                    if (mon_transfer_count[iter_var1] != eff_golden_tokens[iter_var1][31:0]) begin : count_mismatch
                        $display("[tb_module_wrapper] FAIL ch%0d: count mismatch (got %0d, expected %0d)",
                                 iter_var1, mon_transfer_count[iter_var1],
                                 eff_golden_tokens[iter_var1]);
                        ch_mismatch = ch_mismatch + 1;
                    end

                    // Compare tokens
                    for (iter_var0 = 0;
                         iter_var0 < eff_golden_tokens[iter_var1] && iter_var0 < MAX_TOKENS;
                         iter_var0 = iter_var0 + 1) begin : compare_loop
                        case (iter_var1)
                            0: begin : cmp_ch0
                                if (gen_output_ch[0].active_output.u_monitor.capture_mem[iter_var0]
                                    !== golden_mem_0[iter_var0]) begin : mismatch_ch0
                                    if (ch_mismatch < 10) begin : report_ch0
                                        $display("[tb_module_wrapper] MISMATCH ch0[%0d]: got %h, expected %h",
                                                 iter_var0,
                                                 gen_output_ch[0].active_output.u_monitor.capture_mem[iter_var0],
                                                 golden_mem_0[iter_var0]);
                                    end
                                    ch_mismatch = ch_mismatch + 1;
                                end
                            end
                            1: begin : cmp_ch1
                                if (gen_output_ch[1].active_output.u_monitor.capture_mem[iter_var0]
                                    !== golden_mem_1[iter_var0]) begin : mismatch_ch1
                                    if (ch_mismatch < 10) begin : report_ch1
                                        $display("[tb_module_wrapper] MISMATCH ch1[%0d]: got %h, expected %h",
                                                 iter_var0,
                                                 gen_output_ch[1].active_output.u_monitor.capture_mem[iter_var0],
                                                 golden_mem_1[iter_var0]);
                                    end
                                    ch_mismatch = ch_mismatch + 1;
                                end
                            end
                            2: begin : cmp_ch2
                                if (gen_output_ch[2].active_output.u_monitor.capture_mem[iter_var0]
                                    !== golden_mem_2[iter_var0]) begin : mismatch_ch2
                                    if (ch_mismatch < 10) begin : report_ch2
                                        $display("[tb_module_wrapper] MISMATCH ch2[%0d]: got %h, expected %h",
                                                 iter_var0,
                                                 gen_output_ch[2].active_output.u_monitor.capture_mem[iter_var0],
                                                 golden_mem_2[iter_var0]);
                                    end
                                    ch_mismatch = ch_mismatch + 1;
                                end
                            end
                            3: begin : cmp_ch3
                                if (gen_output_ch[3].active_output.u_monitor.capture_mem[iter_var0]
                                    !== golden_mem_3[iter_var0]) begin : mismatch_ch3
                                    if (ch_mismatch < 10) begin : report_ch3
                                        $display("[tb_module_wrapper] MISMATCH ch3[%0d]: got %h, expected %h",
                                                 iter_var0,
                                                 gen_output_ch[3].active_output.u_monitor.capture_mem[iter_var0],
                                                 golden_mem_3[iter_var0]);
                                    end
                                    ch_mismatch = ch_mismatch + 1;
                                end
                            end
                            default: begin : cmp_default
                                // unreachable
                            end
                        endcase
                    end

                    total_mismatch = total_mismatch + ch_mismatch;

                    if (ch_mismatch == 0) begin : ch_pass
                        $display("[tb_module_wrapper] Output[%0d] PASS: All %0d tokens match",
                                 iter_var1, eff_golden_tokens[iter_var1]);
                    end else begin : ch_fail
                        $display("[tb_module_wrapper] Output[%0d] FAIL: %0d mismatches",
                                 iter_var1, ch_mismatch);
                    end
                end
            end

            // Overall verdict
            if (total_golden == 0) begin : no_golden
                $display("[tb_module_wrapper] WARN: No golden traces; skipping comparison");
                sim_pass = 1'b1;
            end else if (total_mismatch == 0) begin : all_pass
                $display("[tb_module_wrapper] PASS: All %0d output channels verified",
                         NUM_DUT_OUTPUTS);
                sim_pass = 1'b1;
            end else begin : any_fail
                $display("[tb_module_wrapper] FAIL: %0d total mismatches across channels",
                         total_mismatch);
                sim_pass = 1'b0;
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
