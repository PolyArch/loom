// tb_channel_driver.sv -- Token driver that reads a hex trace file and drives
//                         a handshake channel (valid/ready protocol).
//
// Trace file format (one line per token, $readmemh compatible):
//   <data_hex>             -- data only (tag defaults to 0)
//   <data_hex> <tag_hex>   -- data and tag
//
// The driver asserts valid when tokens remain. It advances to the next token
// on a successful transfer (valid && ready). When all tokens have been sent,
// it deasserts valid and asserts the 'done' output.
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

module tb_channel_driver #(
    parameter DATA_WIDTH  = 32,
    parameter TAG_WIDTH   = 4,
    parameter MAX_TOKENS  = 4096,
    parameter TRACE_FILE  = "input_trace.hex"
)(
    input  wire                   clk,
    input  wire                   rst_n,

    // Handshake channel output (source side)
    output reg  [DATA_WIDTH-1:0]  data,
    output reg  [TAG_WIDTH-1:0]   tag,
    output reg                    valid,
    input  wire                   ready,

    // Status
    output reg                    done,
    output reg  [31:0]            token_count
);

    // -------------------------------------------------------------------------
    // Token storage
    // -------------------------------------------------------------------------
    // Combined storage: each entry holds {tag, data} packed together.
    // For the trace file, data occupies the lower DATA_WIDTH bits and tag
    // occupies the next TAG_WIDTH bits.
    localparam ENTRY_WIDTH = DATA_WIDTH + TAG_WIDTH;
    reg [ENTRY_WIDTH-1:0] token_mem [0:MAX_TOKENS-1];

    // Number of valid tokens loaded from the trace file
    integer num_tokens;

    // Current index into the token array
    integer token_idx;

    // -------------------------------------------------------------------------
    // Trace loading
    // -------------------------------------------------------------------------
    initial begin : load_trace
        integer iter_var0;
        // Initialize memory to zero
        for (iter_var0 = 0; iter_var0 < MAX_TOKENS; iter_var0 = iter_var0 + 1) begin : mem_init_loop
            token_mem[iter_var0] = {ENTRY_WIDTH{1'b0}};
        end

        $readmemh(TRACE_FILE, token_mem);

        // Count valid tokens by scanning for the first all-zero entry
        // after loading. This heuristic assumes the trace does not
        // intentionally contain all-zero tokens at the end.
        // A more robust approach would use a separate count file.
        num_tokens = 0;
        for (iter_var0 = 0; iter_var0 < MAX_TOKENS; iter_var0 = iter_var0 + 1) begin : count_loop
            if (token_mem[iter_var0] !== {ENTRY_WIDTH{1'bx}}) begin : count_valid
                num_tokens = iter_var0 + 1;
            end
        end

        $display("[tb_channel_driver] Loaded %0d tokens from %s", num_tokens, TRACE_FILE);
    end

    // -------------------------------------------------------------------------
    // Drive logic
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin : drive_proc
        if (!rst_n) begin : drive_reset
            token_idx   <= 0;
            valid       <= 1'b0;
            data        <= {DATA_WIDTH{1'b0}};
            tag         <= {TAG_WIDTH{1'b0}};
            done        <= 1'b0;
            token_count <= 32'd0;
        end else begin : drive_active
            if (done) begin : drive_done_hold
                valid <= 1'b0;
            end else if (token_idx < num_tokens) begin : drive_send
                // Present current token
                data  <= token_mem[token_idx][DATA_WIDTH-1:0];
                tag   <= token_mem[token_idx][ENTRY_WIDTH-1 -: TAG_WIDTH];
                valid <= 1'b1;

                // Advance on transfer
                if (valid && ready) begin : drive_advance
                    token_count <= token_count + 32'd1;
                    if (token_idx + 1 >= num_tokens) begin : drive_last
                        done  <= 1'b1;
                        valid <= 1'b0;
                    end else begin : drive_next
                        token_idx <= token_idx + 1;
                    end
                end
            end else begin : drive_empty
                // No tokens loaded or already past end
                valid <= 1'b0;
                done  <= 1'b1;
            end
        end
    end

endmodule
