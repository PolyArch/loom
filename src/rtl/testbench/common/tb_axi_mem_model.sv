// tb_axi_mem_model.sv -- Simple AXI4-MM slave memory model for testbenches.
//
// Supports single-beat reads and writes (ARLEN/AWLEN = 0). The memory is
// byte-addressable with configurable size. Read responses can be delayed
// by READ_LATENCY cycles to model realistic memory behaviour.
//
// Provides a pre-load task (preload_from_file) to initialize memory
// contents from a hex file via $readmemh.
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

// Single-beat only: burst length, burst size, and wlast signals are part of
// the standard AXI4 interface but not used in this simplified model.
/* verilator lint_off UNUSEDSIGNAL */

module tb_axi_mem_model #(
    parameter ADDR_WIDTH     = 32,
    parameter DATA_WIDTH     = 64,
    parameter ID_WIDTH       = 4,
    parameter MEM_SIZE_BYTES = 65536,
    parameter READ_LATENCY   = 1
)(
    input  wire                      clk,
    input  wire                      rst_n,

    // AXI4 Write Address channel
    input  wire [ID_WIDTH-1:0]       awid,
    input  wire [ADDR_WIDTH-1:0]     awaddr,
    input  wire [7:0]                awlen,
    input  wire [2:0]                awsize,
    input  wire                      awvalid,
    output reg                       awready,

    // AXI4 Write Data channel
    input  wire [DATA_WIDTH-1:0]     wdata,
    input  wire [DATA_WIDTH/8-1:0]   wstrb,
    input  wire                      wlast,
    input  wire                      wvalid,
    output reg                       wready,

    // AXI4 Write Response channel
    output reg  [ID_WIDTH-1:0]       bid,
    output reg  [1:0]                bresp,
    output reg                       bvalid,
    input  wire                      bready,

    // AXI4 Read Address channel
    input  wire [ID_WIDTH-1:0]       arid,
    input  wire [ADDR_WIDTH-1:0]     araddr,
    input  wire [7:0]                arlen,
    input  wire [2:0]                arsize,
    input  wire                      arvalid,
    output reg                       arready,

    // AXI4 Read Data channel
    output reg  [ID_WIDTH-1:0]       rid,
    output reg  [DATA_WIDTH-1:0]     rdata,
    output reg  [1:0]                rresp,
    output reg                       rlast,
    output reg                       rvalid,
    input  wire                      rready
);

/* verilator lint_on UNUSEDSIGNAL */

    // -------------------------------------------------------------------------
    // Memory array (byte-addressable)
    // -------------------------------------------------------------------------
    localparam DATA_BYTES = DATA_WIDTH / 8;
    reg [7:0] mem [0:MEM_SIZE_BYTES-1];

    // -------------------------------------------------------------------------
    // Read latency pipeline
    // -------------------------------------------------------------------------
    localparam RD_PIPE_DEPTH = (READ_LATENCY > 0) ? READ_LATENCY : 1;
    reg                      rd_pipe_valid [0:RD_PIPE_DEPTH-1];
    reg [ID_WIDTH-1:0]       rd_pipe_id    [0:RD_PIPE_DEPTH-1];
    reg [DATA_WIDTH-1:0]     rd_pipe_data  [0:RD_PIPE_DEPTH-1];

    // -------------------------------------------------------------------------
    // Write state
    // -------------------------------------------------------------------------
    reg                      aw_pending;
    reg [ADDR_WIDTH-1:0]     aw_addr_reg;
    reg [ID_WIDTH-1:0]       aw_id_reg;

    // -------------------------------------------------------------------------
    // Memory initialization
    // -------------------------------------------------------------------------
    initial begin : mem_init
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < MEM_SIZE_BYTES; iter_var0 = iter_var0 + 1) begin : mem_zero_loop
            mem[iter_var0] = 8'h00;
        end
    end

    // Task: pre-load memory from hex file
    // The hex file contains byte values, loaded starting at address 0.
    task preload_from_file;
        input [8*256-1:0] filename;  // up to 256 chars
        begin : preload_body
            $readmemh(filename, mem);
            $display("[tb_axi_mem_model] Pre-loaded memory from file at time %0t", $time);
        end
    endtask

    // -------------------------------------------------------------------------
    // Read pipeline initialization
    // -------------------------------------------------------------------------
    initial begin : pipe_init
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < RD_PIPE_DEPTH; iter_var0 = iter_var0 + 1) begin : pipe_zero_loop
            rd_pipe_valid[iter_var0] = 1'b0;
            rd_pipe_id[iter_var0]    = {ID_WIDTH{1'b0}};
            rd_pipe_data[iter_var0]  = {DATA_WIDTH{1'b0}};
        end
    end

    // -------------------------------------------------------------------------
    // Write address channel handling
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin : aw_proc
        if (!rst_n) begin : aw_reset
            awready     <= 1'b1;
            aw_pending  <= 1'b0;
            aw_addr_reg <= {ADDR_WIDTH{1'b0}};
            aw_id_reg   <= {ID_WIDTH{1'b0}};
        end else begin : aw_active
            if (awvalid && awready) begin : aw_accept
                aw_addr_reg <= awaddr;
                aw_id_reg   <= awid;
                aw_pending  <= 1'b1;
                awready     <= 1'b0;  // Wait for write data
            end
            // Re-enable after write completes
            if (aw_pending && wvalid && wready) begin : aw_rearm
                aw_pending <= 1'b0;
                awready    <= 1'b1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Write data channel handling
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin : wd_proc
        integer iter_var0;
        if (!rst_n) begin : wd_reset
            wready <= 1'b0;
            bvalid <= 1'b0;
            bid    <= {ID_WIDTH{1'b0}};
            bresp  <= 2'b00;
        end else begin : wd_active
            // Accept write data when address is pending
            wready <= aw_pending && !bvalid;

            if (wvalid && wready && aw_pending) begin : wd_write
                // Perform byte-lane writes
                for (iter_var0 = 0; iter_var0 < DATA_BYTES; iter_var0 = iter_var0 + 1) begin : byte_write_loop
                    if (wstrb[iter_var0]) begin : byte_write_enabled
                        if ((aw_addr_reg + iter_var0) < MEM_SIZE_BYTES) begin : byte_in_range
                            mem[aw_addr_reg + iter_var0] <= wdata[iter_var0*8 +: 8];
                        end
                    end
                end
                // Issue write response
                bid    <= aw_id_reg;
                bresp  <= 2'b00;  // OKAY
                bvalid <= 1'b1;
            end

            // Clear write response on acceptance
            if (bvalid && bready) begin : wd_resp_clear
                bvalid <= 1'b0;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Read address channel handling + latency pipeline
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin : rd_proc
        integer iter_var0;
        integer iter_var1;
        if (!rst_n) begin : rd_reset
            arready <= 1'b1;
            rvalid  <= 1'b0;
            rdata   <= {DATA_WIDTH{1'b0}};
            rid     <= {ID_WIDTH{1'b0}};
            rresp   <= 2'b00;
            rlast   <= 1'b0;
            for (iter_var0 = 0; iter_var0 < RD_PIPE_DEPTH; iter_var0 = iter_var0 + 1) begin : rd_pipe_reset_loop
                rd_pipe_valid[iter_var0] <= 1'b0;
            end
        end else begin : rd_active
            // Accept read address when pipeline entry is available
            arready <= !rd_pipe_valid[0] || (RD_PIPE_DEPTH == 1 && rvalid && rready);

            // Stage 0: latch new read request
            if (arvalid && arready) begin : rd_accept
                rd_pipe_valid[0] <= 1'b1;
                rd_pipe_id[0]    <= arid;
                // Assemble data from byte memory
                for (iter_var0 = 0; iter_var0 < DATA_BYTES; iter_var0 = iter_var0 + 1) begin : rd_assemble_loop
                    if ((araddr + iter_var0) < MEM_SIZE_BYTES) begin : rd_byte_in_range
                        rd_pipe_data[0][iter_var0*8 +: 8] <= mem[araddr + iter_var0];
                    end else begin : rd_byte_out_of_range
                        rd_pipe_data[0][iter_var0*8 +: 8] <= 8'h00;
                    end
                end
            end else if (rd_pipe_valid[0] && (RD_PIPE_DEPTH == 1 ? (rvalid && rready) : 1'b1)) begin : rd_stage0_clear
                // Pipeline advances or single-stage clears on output handshake
                if (RD_PIPE_DEPTH == 1) begin : rd_single_clear
                    rd_pipe_valid[0] <= 1'b0;
                end
            end

            // Shift pipeline stages (for latency > 1)
            for (iter_var1 = 1; iter_var1 < RD_PIPE_DEPTH; iter_var1 = iter_var1 + 1) begin : rd_pipe_shift_loop
                rd_pipe_valid[iter_var1] <= rd_pipe_valid[iter_var1-1];
                rd_pipe_id[iter_var1]    <= rd_pipe_id[iter_var1-1];
                rd_pipe_data[iter_var1]  <= rd_pipe_data[iter_var1-1];
                if (rd_pipe_valid[iter_var1-1]) begin : rd_pipe_consumed
                    rd_pipe_valid[iter_var1-1] <= 1'b0;
                end
            end

            // Output stage: last pipeline entry drives read data channel
            if (rd_pipe_valid[RD_PIPE_DEPTH-1]) begin : rd_output
                rvalid <= 1'b1;
                rdata  <= rd_pipe_data[RD_PIPE_DEPTH-1];
                rid    <= rd_pipe_id[RD_PIPE_DEPTH-1];
                rresp  <= 2'b00;  // OKAY
                rlast  <= 1'b1;   // Single beat
                rd_pipe_valid[RD_PIPE_DEPTH-1] <= 1'b0;
            end

            // Clear read data on acceptance
            if (rvalid && rready) begin : rd_resp_clear
                rvalid <= 1'b0;
                rlast  <= 1'b0;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Debug: memory read task (for post-sim checking)
    // -------------------------------------------------------------------------
    task read_word;
        input  [ADDR_WIDTH-1:0]  addr;
        output [DATA_WIDTH-1:0]  word_out;
        begin : read_word_body
            integer iter_var0;
            for (iter_var0 = 0; iter_var0 < DATA_BYTES; iter_var0 = iter_var0 + 1) begin : read_byte_loop
                if ((addr + iter_var0) < MEM_SIZE_BYTES) begin : read_in_range
                    word_out[iter_var0*8 +: 8] = mem[addr + iter_var0];
                end else begin : read_out_of_range
                    word_out[iter_var0*8 +: 8] = 8'h00;
                end
            end
        end
    endtask

    // Task: write a word to memory (for programmatic initialization)
    task write_word;
        input [ADDR_WIDTH-1:0]  addr;
        input [DATA_WIDTH-1:0]  word_in;
        begin : write_word_body
            integer iter_var0;
            for (iter_var0 = 0; iter_var0 < DATA_BYTES; iter_var0 = iter_var0 + 1) begin : write_byte_loop
                if ((addr + iter_var0) < MEM_SIZE_BYTES) begin : write_in_range
                    mem[addr + iter_var0] = word_in[iter_var0*8 +: 8];
                end
            end
        end
    endtask

endmodule
