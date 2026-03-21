// tb_config_loader.sv -- Configuration word loader for DUT config interface.
//
// Reads configuration words from a hex file via $readmemh. After reset
// deasserts, drives config words one per cycle through a valid/ready
// handshake, asserting cfg_last on the final word. Asserts config_done
// when all words have been accepted by the DUT.
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

module tb_config_loader #(
    parameter MAX_CONFIG_WORDS = 1024,
    parameter CONFIG_WIDTH     = 32,
    parameter CONFIG_FILE      = "config.hex"
)(
    input  wire                      clk,
    input  wire                      rst_n,

    // Config bus output
    output reg  [CONFIG_WIDTH-1:0]   cfg_wdata,
    output reg                       cfg_valid,
    output reg                       cfg_last,
    input  wire                      cfg_ready,

    // Status
    output reg                       config_done
);

    // -------------------------------------------------------------------------
    // Config word storage
    // -------------------------------------------------------------------------
    reg [CONFIG_WIDTH-1:0] config_mem [0:MAX_CONFIG_WORDS-1];

    // Number of valid config words
    integer num_words;

    // Current word index
    integer word_idx;

    // -------------------------------------------------------------------------
    // Load config words from hex file
    // -------------------------------------------------------------------------
    initial begin : load_config
        integer iter_var0;
        // Initialize memory
        for (iter_var0 = 0; iter_var0 < MAX_CONFIG_WORDS; iter_var0 = iter_var0 + 1) begin : mem_init_loop
            config_mem[iter_var0] = {CONFIG_WIDTH{1'b0}};
        end

        $readmemh(CONFIG_FILE, config_mem);

        // Count words (scan for last non-zero or non-x entry)
        num_words = 0;
        for (iter_var0 = 0; iter_var0 < MAX_CONFIG_WORDS; iter_var0 = iter_var0 + 1) begin : count_loop
            if (config_mem[iter_var0] !== {CONFIG_WIDTH{1'bx}}) begin : count_valid
                num_words = iter_var0 + 1;
            end
        end

        $display("[tb_config_loader] Loaded %0d config words from %s", num_words, CONFIG_FILE);
    end

    // -------------------------------------------------------------------------
    // Drive logic -- present one config word per cycle, advance on transfer
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin : drive_proc
        if (!rst_n) begin : drive_reset
            word_idx    <= 0;
            cfg_wdata   <= {CONFIG_WIDTH{1'b0}};
            cfg_valid   <= 1'b0;
            cfg_last    <= 1'b0;
            config_done <= 1'b0;
        end else begin : drive_active
            if (config_done) begin : drive_done_hold
                cfg_valid <= 1'b0;
                cfg_last  <= 1'b0;
            end else if (word_idx < num_words) begin : drive_send
                cfg_wdata <= config_mem[word_idx];
                cfg_valid <= 1'b1;
                cfg_last  <= (word_idx == num_words - 1) ? 1'b1 : 1'b0;

                // Advance on transfer
                if (cfg_valid && cfg_ready) begin : drive_advance
                    if (word_idx + 1 >= num_words) begin : drive_last_accepted
                        config_done <= 1'b1;
                        cfg_valid   <= 1'b0;
                        cfg_last    <= 1'b0;
                        $display("[tb_config_loader] All %0d config words delivered at time %0t",
                                 num_words, $time);
                    end else begin : drive_next
                        word_idx <= word_idx + 1;
                    end
                end
            end else begin : drive_no_config
                // No config words loaded
                cfg_valid   <= 1'b0;
                cfg_last    <= 1'b0;
                config_done <= 1'b1;
            end
        end
    end

endmodule
