// tb_channel_driver.sv -- Token driver from hex trace file.
//
// Trace format: one hex value per line representing the packed token
// {tag, data} where tag occupies the upper TAG_WIDTH bits and data the
// lower DATA_WIDTH bits.  When TAG_WIDTH == 0, each line is just data.
//
// The number of tokens is provided explicitly via the NUM_TOKENS parameter
// (not auto-detected from the file).
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

module tb_channel_driver #(
    parameter DATA_WIDTH  = 32,
    parameter TAG_WIDTH   = 0,
    parameter NUM_TOKENS  = 0,       // Explicit token count (0 = no tokens)
    parameter MAX_TOKENS  = 4096,
    parameter TRACE_FILE  = "input_trace.hex"
)(
    input  wire                   clk,
    input  wire                   rst_n,

    // Handshake channel output (source side)
    output reg  [DATA_WIDTH-1:0]  data,
    output reg  [TAG_WIDTH > 0 ? TAG_WIDTH-1 : 0 : 0] tag,
    output reg                    valid,
    input  wire                   ready,

    // Status
    output reg                    done,
    output reg  [31:0]            token_count
);

    // -------------------------------------------------------------------------
    // Token storage -- packed {tag, data} per entry
    // -------------------------------------------------------------------------
    localparam TAG_W = (TAG_WIDTH > 0) ? TAG_WIDTH : 1;
    localparam ENTRY_WIDTH = DATA_WIDTH + TAG_W;
    reg [ENTRY_WIDTH-1:0] token_mem [0:MAX_TOKENS-1];

    // Current index into the token array
    integer token_idx;

    // Effective number of tokens
    integer eff_num_tokens;

    // -------------------------------------------------------------------------
    // Trace loading
    // -------------------------------------------------------------------------
    initial begin : load_trace
        eff_num_tokens = NUM_TOKENS;
        if (eff_num_tokens > MAX_TOKENS) begin : clamp_tokens
            eff_num_tokens = MAX_TOKENS;
        end : clamp_tokens

        if (eff_num_tokens > 0) begin : do_load
            $readmemh(TRACE_FILE, token_mem, 0, eff_num_tokens - 1);
        end : do_load

        $display("[tb_channel_driver] %0d tokens from %s", eff_num_tokens, TRACE_FILE);
    end

    // -------------------------------------------------------------------------
    // Drive logic
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin : drive_proc
        if (!rst_n) begin : drive_reset
            token_idx   <= 0;
            valid       <= 1'b0;
            data        <= '0;
            tag         <= '0;
            done        <= (eff_num_tokens == 0) ? 1'b1 : 1'b0;
            token_count <= 32'd0;
        end else begin : drive_active
            if (done) begin : drive_done_hold
                valid <= 1'b0;
            end else if (token_idx < eff_num_tokens) begin : drive_send
                data  <= token_mem[token_idx][DATA_WIDTH-1:0];
                if (TAG_WIDTH > 0) begin : drive_tag
                    tag <= token_mem[token_idx][ENTRY_WIDTH-1 -: TAG_W];
                end : drive_tag
                valid <= 1'b1;

                if (valid && ready) begin : drive_advance
                    token_count <= token_count + 32'd1;
                    if (token_idx + 1 >= eff_num_tokens) begin : drive_last
                        done  <= 1'b1;
                        valid <= 1'b0;
                    end else begin : drive_next
                        token_idx <= token_idx + 1;
                        // Look ahead: present next token immediately so
                        // combinational DUTs don't see stale data for
                        // one extra cycle after handshake completion.
                        data <= token_mem[token_idx + 1][DATA_WIDTH-1:0];
                        if (TAG_WIDTH > 0) begin : drive_tag_next
                            tag <= token_mem[token_idx + 1][ENTRY_WIDTH-1 -: TAG_W];
                        end : drive_tag_next
                    end
                end
            end else begin : drive_empty
                valid <= 1'b0;
                done  <= 1'b1;
            end
        end
    end

endmodule
