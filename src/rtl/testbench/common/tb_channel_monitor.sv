// tb_channel_monitor.sv -- Transfer capture for a handshake channel.
//
// On each successful transfer (valid && ready), records the packed
// token {tag, data} as one hex value per line to the output trace file.
// This format is directly round-trippable into tb_channel_driver and
// tb_module_wrapper golden comparison.
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

module tb_channel_monitor #(
    parameter DATA_WIDTH  = 32,
    parameter TAG_WIDTH   = 0,
    parameter MAX_TOKENS  = 4096,
    parameter TRACE_FILE  = "output_trace.hex"
)(
    input  wire                   clk,
    input  wire                   rst_n,

    // Handshake channel input (sink side -- observe only)
    input  wire [DATA_WIDTH-1:0]  data,
    input  wire [TAG_WIDTH > 0 ? TAG_WIDTH-1 : 0 : 0] tag,
    input  wire                   valid,
    input  wire                   ready,

    // Status
    output reg  [31:0]            transfer_count
);

    // -------------------------------------------------------------------------
    // Internal storage for captured transfers
    // -------------------------------------------------------------------------
    localparam TAG_W = (TAG_WIDTH > 0) ? TAG_WIDTH : 1;
    localparam ENTRY_WIDTH = DATA_WIDTH + TAG_W;
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
            $display("[tb_channel_monitor] ERROR: Cannot open %s", TRACE_FILE);
            file_open = 1'b0;
        end else begin : file_open_ok
            file_open = 1'b1;
        end
    end

    // -------------------------------------------------------------------------
    // Capture logic -- write packed {tag, data} as one hex value per line
    // -------------------------------------------------------------------------
    /* verilator lint_off BLKSEQ */
    always @(posedge clk or negedge rst_n) begin : capture_proc
        if (!rst_n) begin : capture_reset
            transfer_count <= 32'd0;
        end else begin : capture_active
            if (valid && ready) begin : capture_transfer
                if (file_open) begin : capture_write
                    // Write packed token as single hex value per line
                    if (TAG_WIDTH > 0) begin : capture_with_tag
                        $fwrite(fd, "%h\n", {tag, data});
                    end else begin : capture_no_tag
                        $fwrite(fd, "%h\n", data);
                    end
                    $fflush(fd);
                end

                // Store in memory buffer
                if (transfer_count < MAX_TOKENS) begin : capture_store
                    if (TAG_WIDTH > 0) begin : store_tagged
                        capture_mem[transfer_count] = {tag, data};
                    end else begin : store_untagged
                        capture_mem[transfer_count] = {{TAG_W{1'b0}}, data};
                    end
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
        end
    end

endmodule
