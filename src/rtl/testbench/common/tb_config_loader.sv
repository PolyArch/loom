// tb_config_loader.sv -- Configuration word loader for DUT config interface.
//
// Reads configuration words from a hex file via $readmemh. The number of
// words is provided explicitly via NUM_WORDS (not auto-detected). After
// reset deasserts, drives config words one per cycle through valid/ready
// handshake, asserting cfg_last on the final word.
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

module tb_config_loader #(
    parameter NUM_WORDS        = 0,       // Explicit word count
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

    // Current word index
    integer word_idx;

    // Effective number of words
    integer eff_num_words;

    // -------------------------------------------------------------------------
    // Load config words from hex file
    // -------------------------------------------------------------------------
    initial begin : load_config
        eff_num_words = NUM_WORDS;
        if (eff_num_words > MAX_CONFIG_WORDS) begin : clamp_words
            eff_num_words = MAX_CONFIG_WORDS;
        end : clamp_words

        if (eff_num_words > 0) begin : do_load
            $readmemh(CONFIG_FILE, config_mem, 0, eff_num_words - 1);
        end : do_load

        $display("[tb_config_loader] %0d config words from %s", eff_num_words, CONFIG_FILE);
    end

    // -------------------------------------------------------------------------
    // Drive logic
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin : drive_proc
        if (!rst_n) begin : drive_reset
            word_idx    <= 0;
            cfg_wdata   <= '0;
            cfg_valid   <= 1'b0;
            cfg_last    <= 1'b0;
            config_done <= (eff_num_words == 0) ? 1'b1 : 1'b0;
        end else begin : drive_active
            if (config_done) begin : drive_done_hold
                cfg_valid <= 1'b0;
                cfg_last  <= 1'b0;
            end else if (word_idx < eff_num_words) begin : drive_send
                cfg_wdata <= config_mem[word_idx];
                cfg_valid <= 1'b1;
                cfg_last  <= (word_idx == eff_num_words - 1) ? 1'b1 : 1'b0;

                if (cfg_valid && cfg_ready) begin : drive_advance
                    if (word_idx + 1 >= eff_num_words) begin : drive_last_accepted
                        config_done <= 1'b1;
                        cfg_valid   <= 1'b0;
                        cfg_last    <= 1'b0;
                    end else begin : drive_next
                        word_idx <= word_idx + 1;
                    end
                end
            end else begin : drive_no_config
                cfg_valid   <= 1'b0;
                cfg_last    <= 1'b0;
                config_done <= 1'b1;
            end
        end
    end

endmodule
