// tb_module_wrapper.sv -- Generic testbench wrapper template.
//
// Wires together shared testbench infrastructure for testing any DUT:
//   - tb_clk_rst_gen, tb_config_loader, tb_channel_driver,
//     tb_channel_monitor, tb_backpressure_gen
//
// Supports up to NUM_DUT_INPUTS input channels and NUM_DUT_OUTPUTS output
// channels via generate loops. Each channel reads its own plusarg for
// trace file path and token count.
//
// The DUT is instantiated via the `DUT_MODULE` macro, which must be
// defined at compile time (e.g., verilator +define+DUT_MODULE=fabric_top_xyz
// or VCS +define+DUT_MODULE=...).  If not defined, the wrapper falls back
// to a loopback stub for self-testing of the TB infrastructure.
//
// Trace format: all traces use packed hex (one value per line).
//   - Input trace: {tag, data} per line (or just data if TAG_WIDTH==0)
//   - Output trace: same format
//   - Golden trace: same format
//   - Config trace: one 32-bit word per line
//
// Runtime-configurable via Verilator/VCS plusargs:
//   +INPUT_TRACE_0=path  +INPUT_TRACE_1=path  ... (up to NUM_DUT_INPUTS-1)
//   +OUTPUT_TRACE_0=path +OUTPUT_TRACE_1=path ... (up to NUM_DUT_OUTPUTS-1)
//   +GOLDEN_TRACE_0=path +GOLDEN_TRACE_1=path ... (up to NUM_DUT_OUTPUTS-1)
//   +CONFIG_FILE=path     +NUM_INPUT_TOKENS=N     +GOLDEN_TOKENS=N
//   +NUM_CONFIG_WORDS=N   +SIM_TIMEOUT_CYCLES=N
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

    // Multi-channel support: up to 4 inputs and 4 outputs
    parameter NUM_DUT_INPUTS     = 1;
    parameter NUM_DUT_OUTPUTS    = 1;

    // Explicit token/word counts (no auto-detection)
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
    // Plusarg overrides -- runtime configuration via +NAME=value
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
    integer          plusarg_num_input_tokens;
    integer          plusarg_golden_tokens;
    integer          plusarg_num_config_words;
    integer          plusarg_sim_timeout;

    // Effective values after plusarg resolution
    reg [4096*8-1:0] eff_input_trace  [0:3];
    reg [4096*8-1:0] eff_output_trace [0:3];
    reg [4096*8-1:0] eff_golden_trace [0:3];
    reg [4096*8-1:0] eff_config_file;
    integer          eff_num_input_tokens;
    integer          eff_golden_tokens;
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

        // Config file plusarg
        eff_config_file = CONFIG_FILE;
        if ($value$plusargs("CONFIG_FILE=%s", plusarg_config_file)) begin : use_config_file
            eff_config_file = plusarg_config_file;
        end

        // Integer plusargs
        eff_num_input_tokens = NUM_INPUT_TOKENS;
        if ($value$plusargs("NUM_INPUT_TOKENS=%d", plusarg_num_input_tokens)) begin : use_input_tokens
            eff_num_input_tokens = plusarg_num_input_tokens;
        end

        eff_golden_tokens = GOLDEN_TOKENS;
        if ($value$plusargs("GOLDEN_TOKENS=%d", plusarg_golden_tokens)) begin : use_golden_tokens
            eff_golden_tokens = plusarg_golden_tokens;
        end

        eff_num_config_words = NUM_CONFIG_WORDS;
        if ($value$plusargs("NUM_CONFIG_WORDS=%d", plusarg_num_config_words)) begin : use_config_words
            eff_num_config_words = plusarg_num_config_words;
        end

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
    //
    // The config loader reads its own plusargs (+CONFIG_FILE, +NUM_CONFIG_WORDS)
    // so it uses the runtime-resolved values rather than compile-time params.
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
    // Multi-channel wires -- sized for up to 4 channels
    // =========================================================================
    localparam TAG_W = (TAG_WIDTH > 0) ? TAG_WIDTH : 1;
    localparam MAX_CHANNELS = 4;

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
    // Each driver reads its own plusarg for the trace file at runtime.
    // The driver's TRACE_FILE parameter is the compile-time default;
    // the plusarg override happens inside the driver's own $readmemh call
    // which we replace: the driver now reads from its parent's eff_* arrays
    // via an explicit initial-block override.
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
                // using the plusarg-resolved values from the wrapper.
                initial begin : override_driver
                    #0; // ensure plusarg_init has executed
                    u_driver.eff_num_tokens = eff_num_input_tokens;
                    if (eff_num_input_tokens > 0) begin : do_reload
                        $readmemh(eff_input_trace[gi], u_driver.token_mem,
                                  0, eff_num_input_tokens - 1);
                        $display("[tb_module_wrapper] Driver[%0d]: %0d tokens from %s",
                                 gi, eff_num_input_tokens, eff_input_trace[gi]);
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
    //
    // Each monitor writes captured tokens to its own trace file. The
    // TRACE_FILE parameter is the compile-time default; the runtime
    // override from plusargs is applied via hierarchical reference.
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
    //
    // Reload config from plusarg-resolved path and word count.
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
    // The DUT_MODULE macro selects the generated fabric_top_* module.
    // Define it at compile time:
    //   verilator +define+DUT_MODULE=fabric_top_test_fifo_depth4
    //   vcs       +define+DUT_MODULE=fabric_top_test_fifo_depth4
    //
    // The generated top module ports follow a standard convention:
    //   clk, rst_n,
    //   mod_in0, mod_in0_valid, mod_in0_ready,
    //   mod_in1, mod_in1_valid, mod_in1_ready, ... (up to NUM_DUT_INPUTS-1)
    //   mod_out0, mod_out0_valid, mod_out0_ready,
    //   mod_out1, mod_out1_valid, mod_out1_ready, ... (up to NUM_DUT_OUTPUTS-1)
    //   cfg_valid, cfg_wdata, cfg_last, cfg_ready
    //
    // We connect using explicit port wiring for up to 4 channels.
    // Unused channels' ready signals are tied high; unused valid/data are
    // left unconnected (the DUT does not produce them).
    // =========================================================================

`ifdef DUT_MODULE

    // 1-input, 1-output DUT (most common: FIFOs, add_tag, del_tag, etc.)
    `ifdef DUT_PORTS_1IN_1OUT
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

    // 2-input, 2-output DUT (spatial switches, 2x2 grids)
    `elsif DUT_PORTS_2IN_2OUT
    `DUT_MODULE u_dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .mod_in0        (drv_data[0]),
        .mod_in0_valid  (drv_valid[0]),
        .mod_in0_ready  (drv_ready[0]),
        .mod_in1        (drv_data[1]),
        .mod_in1_valid  (drv_valid[1]),
        .mod_in1_ready  (drv_ready[1]),
        .mod_out0       (mon_data[0]),
        .mod_out0_valid (mon_valid[0]),
        .mod_out0_ready (mon_ready[0]),
        .mod_out1       (mon_data[1]),
        .mod_out1_valid (mon_valid[1]),
        .mod_out1_ready (mon_ready[1]),
        .cfg_valid      (cfg_valid),
        .cfg_wdata      (cfg_wdata),
        .cfg_last       (cfg_last),
        .cfg_ready      (cfg_ready)
    );

    // 3-input, 1-output DUT (e.g., chess_2x2 with 3 inputs, 1 output)
    `elsif DUT_PORTS_3IN_1OUT
    `DUT_MODULE u_dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .mod_in0        (drv_data[0]),
        .mod_in0_valid  (drv_valid[0]),
        .mod_in0_ready  (drv_ready[0]),
        .mod_in1        (drv_data[1]),
        .mod_in1_valid  (drv_valid[1]),
        .mod_in1_ready  (drv_ready[1]),
        .mod_in2        (drv_data[2]),
        .mod_in2_valid  (drv_valid[2]),
        .mod_in2_ready  (drv_ready[2]),
        .mod_out0       (mon_data[0]),
        .mod_out0_valid (mon_valid[0]),
        .mod_out0_ready (mon_ready[0]),
        .cfg_valid      (cfg_valid),
        .cfg_wdata      (cfg_wdata),
        .cfg_last       (cfg_last),
        .cfg_ready      (cfg_ready)
    );

    // 4-input, 4-output DUT (maximum supported)
    `elsif DUT_PORTS_4IN_4OUT
    `DUT_MODULE u_dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .mod_in0        (drv_data[0]),
        .mod_in0_valid  (drv_valid[0]),
        .mod_in0_ready  (drv_ready[0]),
        .mod_in1        (drv_data[1]),
        .mod_in1_valid  (drv_valid[1]),
        .mod_in1_ready  (drv_ready[1]),
        .mod_in2        (drv_data[2]),
        .mod_in2_valid  (drv_valid[2]),
        .mod_in2_ready  (drv_ready[2]),
        .mod_in3        (drv_data[3]),
        .mod_in3_valid  (drv_valid[3]),
        .mod_in3_ready  (drv_ready[3]),
        .mod_out0       (mon_data[0]),
        .mod_out0_valid (mon_valid[0]),
        .mod_out0_ready (mon_ready[0]),
        .mod_out1       (mon_data[1]),
        .mod_out1_valid (mon_valid[1]),
        .mod_out1_ready (mon_ready[1]),
        .mod_out2       (mon_data[2]),
        .mod_out2_valid (mon_valid[2]),
        .mod_out2_ready (mon_ready[2]),
        .mod_out3       (mon_data[3]),
        .mod_out3_valid (mon_valid[3]),
        .mod_out3_ready (mon_ready[3]),
        .cfg_valid      (cfg_valid),
        .cfg_wdata      (cfg_wdata),
        .cfg_last       (cfg_last),
        .cfg_ready      (cfg_ready)
    );

    // 2-input, 1-output DUT (PEs with 2 operands, single output)
    `elsif DUT_PORTS_2IN_1OUT
    `DUT_MODULE u_dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .mod_in0        (drv_data[0]),
        .mod_in0_valid  (drv_valid[0]),
        .mod_in0_ready  (drv_ready[0]),
        .mod_in1        (drv_data[1]),
        .mod_in1_valid  (drv_valid[1]),
        .mod_in1_ready  (drv_ready[1]),
        .mod_out0       (mon_data[0]),
        .mod_out0_valid (mon_valid[0]),
        .mod_out0_ready (mon_ready[0]),
        .cfg_valid      (cfg_valid),
        .cfg_wdata      (cfg_wdata),
        .cfg_last       (cfg_last),
        .cfg_ready      (cfg_ready)
    );

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
    `endif

    // Tie off unused input channel ready signals for DUT ports not connected
    generate
        genvar gtr;
        for (gtr = 0; gtr < MAX_CHANNELS; gtr = gtr + 1) begin : gen_tieoff_ready
            if (gtr >= NUM_DUT_INPUTS) begin : tieoff_drv_ready
                assign drv_ready[gtr] = 1'b1;
            end
        end
    endgenerate

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
    // Golden trace comparison (channel 0 -- primary output)
    //
    // Uses eff_golden_tokens / eff_golden_trace[0] resolved from plusargs.
    // The golden trace is loaded after plusargs are resolved (deferred to a
    // second initial block that runs after time 0 via #0).
    // =========================================================================
    localparam ENTRY_WIDTH = DATA_WIDTH + TAG_W;
    reg [ENTRY_WIDTH-1:0] golden_mem [0:MAX_TOKENS-1];
    integer mismatch_count;

    initial begin : golden_load
        #0; // ensure plusarg_init has executed
        if (eff_golden_tokens > 0) begin : do_load_golden
            $readmemh(eff_golden_trace[0], golden_mem, 0, eff_golden_tokens - 1);
            $display("[tb_module_wrapper] Loaded %0d golden tokens from %s",
                     eff_golden_tokens, eff_golden_trace[0]);
        end else begin : no_golden_load
            $display("[tb_module_wrapper] No golden tokens specified");
        end
    end

    // =========================================================================
    // Simulation timeout and done detection
    //
    // Uses eff_golden_tokens and eff_sim_timeout for runtime flexibility.
    // Done when all input drivers have finished and the primary output
    // channel (channel 0) has received enough golden tokens.
    // =========================================================================
    integer cycle_count;
    reg     sim_done;
    reg     sim_pass;

    // All input drivers done
    wire all_drivers_done;
    assign all_drivers_done = drv_done[0] & drv_done[1] & drv_done[2] & drv_done[3];

    always @(posedge clk or negedge rst_n) begin : timeout_proc
        if (!rst_n) begin : timeout_reset
            cycle_count <= 0;
            sim_done    <= 1'b0;
        end else begin : timeout_active
            cycle_count <= cycle_count + 1;

            if (!sim_done && config_done && all_drivers_done) begin : check_done
                if (eff_golden_tokens > 0) begin : check_golden_done
                    if (mon_transfer_count[0] >= eff_golden_tokens[31:0]) begin : all_out
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
    // Final comparison and verdict (compares channel 0 output against golden)
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

            $display("[tb_module_wrapper] Output[0]: %0d transfers, Golden: %0d tokens",
                     mon_transfer_count[0], eff_golden_tokens);

            if (eff_golden_tokens > 0) begin : compare_with_golden
                if (mon_transfer_count[0] != eff_golden_tokens[31:0]) begin : count_mismatch
                    $display("[tb_module_wrapper] FAIL: count mismatch (got %0d, expected %0d)",
                             mon_transfer_count[0], eff_golden_tokens);
                    mismatch_count = mismatch_count + 1;
                end

                for (iter_var0 = 0; iter_var0 < eff_golden_tokens && iter_var0 < MAX_TOKENS;
                     iter_var0 = iter_var0 + 1) begin : compare_loop
                    if (gen_output_ch[0].active_output.u_monitor.capture_mem[iter_var0] !== golden_mem[iter_var0]) begin : token_mismatch
                        if (mismatch_count < 10) begin : report_mismatch
                            $display("[tb_module_wrapper] MISMATCH[%0d]: got %h, expected %h",
                                     iter_var0,
                                     gen_output_ch[0].active_output.u_monitor.capture_mem[iter_var0],
                                     golden_mem[iter_var0]);
                        end
                        mismatch_count = mismatch_count + 1;
                    end
                end

                if (mismatch_count == 0) begin : pass_verdict
                    $display("[tb_module_wrapper] PASS: All %0d tokens match",
                             eff_golden_tokens);
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
