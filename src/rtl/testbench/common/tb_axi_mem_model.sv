// tb_axi_mem_model.sv -- Simple AXI4-MM slave memory model for testbenches.
//
// Supports single-beat reads and writes (ARLEN/AWLEN = 0). The memory is
// byte-addressable with configurable size. Read responses can be delayed
// by READ_LATENCY cycles.
//
// The read path protects against overwriting a stalled rvalid response:
// arready is deasserted when the output register holds an undrained response.
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

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

    localparam DATA_BYTES = DATA_WIDTH / 8;
    reg [7:0] mem [0:MEM_SIZE_BYTES-1];

    // -------------------------------------------------------------------------
    // Memory initialization
    // -------------------------------------------------------------------------
    initial begin : mem_init
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < MEM_SIZE_BYTES; iter_var0 = iter_var0 + 1) begin : mem_zero
            mem[iter_var0] = 8'h00;
        end : mem_zero
    end

    task preload_from_file;
        input [8*256-1:0] filename;
        begin : preload_body
            $readmemh(filename, mem);
        end
    endtask

    // -------------------------------------------------------------------------
    // Write channels (AW + W -> B)
    // -------------------------------------------------------------------------
    reg                      aw_pending;
    reg [ADDR_WIDTH-1:0]     aw_addr_reg;
    reg [ID_WIDTH-1:0]       aw_id_reg;

    always @(posedge clk or negedge rst_n) begin : aw_proc
        if (!rst_n) begin : aw_reset
            awready     <= 1'b1;
            aw_pending  <= 1'b0;
            aw_addr_reg <= '0;
            aw_id_reg   <= '0;
        end else begin : aw_active
            if (awvalid && awready) begin : aw_accept
                aw_addr_reg <= awaddr;
                aw_id_reg   <= awid;
                aw_pending  <= 1'b1;
                awready     <= 1'b0;
            end
            if (aw_pending && wvalid && wready) begin : aw_rearm
                aw_pending <= 1'b0;
                awready    <= 1'b1;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin : wd_proc
        integer iter_var0;
        if (!rst_n) begin : wd_reset
            wready <= 1'b0;
            bvalid <= 1'b0;
            bid    <= '0;
            bresp  <= 2'b00;
        end else begin : wd_active
            wready <= aw_pending && !bvalid;

            if (wvalid && wready && aw_pending) begin : wd_write
                for (iter_var0 = 0; iter_var0 < DATA_BYTES; iter_var0 = iter_var0 + 1) begin : byte_wr
                    if (wstrb[iter_var0]) begin : strb_en
                        if ((aw_addr_reg + iter_var0) < MEM_SIZE_BYTES) begin : in_range
                            mem[aw_addr_reg + iter_var0] <= wdata[iter_var0*8 +: 8];
                        end : in_range
                    end : strb_en
                end : byte_wr
                bid    <= aw_id_reg;
                bresp  <= 2'b00;
                bvalid <= 1'b1;
            end

            if (bvalid && bready) begin : wd_resp_clear
                bvalid <= 1'b0;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Read channel with backpressure protection
    //
    // A pending read request is latched into rd_pending. The read data is
    // assembled from memory and placed into output registers. arready is
    // deasserted while a response is pending (either in the latency pipeline
    // or stalled at the output waiting for rready).
    // -------------------------------------------------------------------------
    reg                      rd_pending;
    reg [ID_WIDTH-1:0]       rd_pending_id;
    reg [DATA_WIDTH-1:0]     rd_pending_data;
    reg [$clog2(READ_LATENCY > 0 ? READ_LATENCY : 1)-1:0] rd_latency_cnt;
    reg                      rd_output_valid;  // output register has undrained data

    always @(posedge clk or negedge rst_n) begin : rd_proc
        integer iter_var0;
        if (!rst_n) begin : rd_reset
            arready        <= 1'b1;
            rvalid         <= 1'b0;
            rdata          <= '0;
            rid            <= '0;
            rresp          <= 2'b00;
            rlast          <= 1'b0;
            rd_pending     <= 1'b0;
            rd_pending_id  <= '0;
            rd_pending_data <= '0;
            rd_latency_cnt <= '0;
            rd_output_valid <= 1'b0;
        end else begin : rd_active
            // Drain output on rready
            if (rvalid && rready) begin : rd_drain
                rvalid          <= 1'b0;
                rlast           <= 1'b0;
                rd_output_valid <= 1'b0;
            end

            // Accept new read address only when no pending request and output is free
            arready <= !rd_pending && !rd_output_valid && !(rvalid && !rready);

            if (arvalid && arready) begin : rd_accept
                rd_pending    <= 1'b1;
                rd_pending_id <= arid;
                // Assemble data from byte memory
                for (iter_var0 = 0; iter_var0 < DATA_BYTES; iter_var0 = iter_var0 + 1) begin : rd_byte
                    if ((araddr + iter_var0) < MEM_SIZE_BYTES) begin : rd_in_range
                        rd_pending_data[iter_var0*8 +: 8] <= mem[araddr + iter_var0];
                    end else begin : rd_out_range
                        rd_pending_data[iter_var0*8 +: 8] <= 8'h00;
                    end : rd_out_range
                end : rd_byte
                rd_latency_cnt <= (READ_LATENCY > 1) ? READ_LATENCY[($clog2(READ_LATENCY > 0 ? READ_LATENCY : 1))-1:0] - 1 : '0;
            end

            // Count down latency
            if (rd_pending) begin : rd_countdown
                if (rd_latency_cnt > 0) begin : rd_wait
                    rd_latency_cnt <= rd_latency_cnt - 1;
                end else begin : rd_deliver
                    // Move to output register (only if output is free)
                    if (!rd_output_valid && !(rvalid && !rready)) begin : rd_to_output
                        rvalid          <= 1'b1;
                        rdata           <= rd_pending_data;
                        rid             <= rd_pending_id;
                        rresp           <= 2'b00;
                        rlast           <= 1'b1;
                        rd_pending      <= 1'b0;
                        rd_output_valid <= 1'b1;
                    end : rd_to_output
                end : rd_deliver
            end : rd_countdown
        end
    end

    // -------------------------------------------------------------------------
    // Debug tasks
    // -------------------------------------------------------------------------
    task read_word;
        input  [ADDR_WIDTH-1:0]  addr;
        output [DATA_WIDTH-1:0]  word_out;
        begin : read_word_body
            integer iter_var0;
            for (iter_var0 = 0; iter_var0 < DATA_BYTES; iter_var0 = iter_var0 + 1) begin : rd_loop
                if ((addr + iter_var0) < MEM_SIZE_BYTES) begin : in_range
                    word_out[iter_var0*8 +: 8] = mem[addr + iter_var0];
                end else begin : out_range
                    word_out[iter_var0*8 +: 8] = 8'h00;
                end
            end : rd_loop
        end
    endtask

    task write_word;
        input [ADDR_WIDTH-1:0]  addr;
        input [DATA_WIDTH-1:0]  word_in;
        begin : write_word_body
            integer iter_var0;
            for (iter_var0 = 0; iter_var0 < DATA_BYTES; iter_var0 = iter_var0 + 1) begin : wr_loop
                if ((addr + iter_var0) < MEM_SIZE_BYTES) begin : in_range
                    mem[addr + iter_var0] = word_in[iter_var0*8 +: 8];
                end
            end : wr_loop
        end
    endtask

endmodule
