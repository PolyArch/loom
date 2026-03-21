// tb_channel_monitor.sv -- Transfer capture for a handshake channel.
//
// On each successful transfer (valid && ready), records {data, tag} to an
// output trace file and increments the transfer counter. The output format
// is one hex entry per line, compatible with $readmemh for golden comparison.
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

module tb_channel_monitor #(
    parameter DATA_WIDTH  = 32,
    parameter TAG_WIDTH   = 4,
    parameter MAX_TOKENS  = 4096,
    parameter TRACE_FILE  = "output_trace.hex"
)(
    input  wire                   clk,
    input  wire                   rst_n,

    // Handshake channel input (sink side -- observe only)
    input  wire [DATA_WIDTH-1:0]  data,
    input  wire [TAG_WIDTH-1:0]   tag,
    input  wire                   valid,
    input  wire                   ready,

    // Status
    output reg  [31:0]            transfer_count
);

    // -------------------------------------------------------------------------
    // Internal storage for captured transfers (in-memory buffer for post-sim
    // comparison, accessed by tb_module_wrapper via hierarchical reference)
    // -------------------------------------------------------------------------
    localparam ENTRY_WIDTH = DATA_WIDTH + TAG_WIDTH;
    reg [ENTRY_WIDTH-1:0] capture_mem [0:MAX_TOKENS-1] /* verilator public */;

    // File descriptor for output trace
    integer fd;
    reg     file_open;

    // -------------------------------------------------------------------------
    // Open output file on simulation start
    // -------------------------------------------------------------------------
    initial begin : file_init
        fd = $fopen(TRACE_FILE, "w");
        if (fd == 0) begin : file_open_fail
            $display("[tb_channel_monitor] ERROR: Cannot open %s for writing", TRACE_FILE);
            file_open = 1'b0;
        end else begin : file_open_ok
            $display("[tb_channel_monitor] Opened %s for writing", TRACE_FILE);
            file_open = 1'b1;
        end
    end

    // -------------------------------------------------------------------------
    // Capture logic
    // -------------------------------------------------------------------------
    // The capture_mem write uses blocking assignment intentionally: it is a
    // testbench memory buffer, not synthesizable flip-flop state.
    /* verilator lint_off BLKSEQ */
    always @(posedge clk or negedge rst_n) begin : capture_proc
        if (!rst_n) begin : capture_reset
            transfer_count <= 32'd0;
        end else begin : capture_active
            if (valid && ready) begin : capture_transfer
                // Write to output file: <tag_hex> <data_hex>
                if (file_open) begin : capture_write
                    if (TAG_WIDTH > 0) begin : capture_with_tag
                        $fwrite(fd, "%h %h\n", tag, data);
                    end else begin : capture_no_tag
                        $fwrite(fd, "%h\n", data);
                    end
                    $fflush(fd);
                end

                // Store in memory buffer (for post-sim comparison)
                if (transfer_count < MAX_TOKENS) begin : capture_store
                    capture_mem[transfer_count] = {tag, data};
                end

                transfer_count <= transfer_count + 32'd1;
            end
        end
    end
    /* verilator lint_on BLKSEQ */

    // -------------------------------------------------------------------------
    // Close file at end of simulation
    // -------------------------------------------------------------------------
    final begin : file_close
        if (file_open) begin : close_fd
            $fclose(fd);
            $display("[tb_channel_monitor] Closed %s after %0d transfers",
                     TRACE_FILE, transfer_count);
        end
    end

endmodule
