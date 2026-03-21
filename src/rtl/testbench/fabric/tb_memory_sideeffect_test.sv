// tb_memory_sideeffect_test.sv -- Write-then-read verification for fabric_memory.
//
// Writes NUM_WORDS sequential data values to sequential addresses via the
// store port, then reads them back via the load port. Compares read data
// against written data and reports PASS/FAIL.
//
// Uses a single region covering the entire scratchpad. Tag=0 for all
// requests (single-lane operation).
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

module tb_memory_sideeffect_test;

    // =========================================================================
    // Parameters
    // =========================================================================
    parameter DATA_WIDTH         = 32;
    parameter TAG_WIDTH          = 4;
    parameter NUM_REGION         = 1;
    parameter SPAD_SIZE_BYTES    = 1024;
    parameter NUM_WORDS          = 16;
    parameter CLK_PERIOD_NS      = 10;
    parameter RST_CYCLES         = 5;
    parameter SIM_TIMEOUT_CYCLES = 20000;

    localparam TAG_W = (TAG_WIDTH > 0) ? TAG_WIDTH : 1;

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
    // DUT: fabric_memory (private, no AXI)
    // =========================================================================
    // Config port
    logic                   cfg_valid;
    logic [31:0]            cfg_wdata;
    logic                   cfg_ready;

    // Load address port
    logic                   load_addr_valid;
    logic                   load_addr_ready;
    logic [DATA_WIDTH-1:0]  load_addr_data;
    logic [TAG_W-1:0]       load_addr_tag;

    // Store address port
    logic                   store_addr_valid;
    logic                   store_addr_ready;
    logic [DATA_WIDTH-1:0]  store_addr_data;
    logic [TAG_W-1:0]       store_addr_tag;

    // Store data port
    logic                   store_data_valid;
    logic                   store_data_ready;
    logic [DATA_WIDTH-1:0]  store_data_data;
    logic [TAG_W-1:0]       store_data_tag;

    // Load data output
    logic                   load_data_valid;
    logic                   load_data_ready;
    logic [DATA_WIDTH-1:0]  load_data_data;
    logic [TAG_W-1:0]       load_data_tag;

    // Load done output
    logic                   load_done_valid;
    logic                   load_done_ready;
    logic [TAG_W-1:0]       load_done_tag;

    // Store done output
    logic                   store_done_valid;
    logic                   store_done_ready;
    logic [TAG_W-1:0]       store_done_tag;

    fabric_memory #(
        .LD_COUNT        (1),
        .ST_COUNT        (1),
        .DATA_WIDTH      (DATA_WIDTH),
        .TAG_WIDTH       (TAG_WIDTH),
        .LSQ_DEPTH       (4),
        .NUM_REGION      (NUM_REGION),
        .IS_PRIVATE      (1'b1),
        .SPAD_SIZE_BYTES (SPAD_SIZE_BYTES),
        .ADDR_WIDTH      (32)
    ) u_dut (
        .clk              (clk),
        .rst_n            (rst_n),
        .cfg_valid        (cfg_valid),
        .cfg_wdata        (cfg_wdata),
        .cfg_ready        (cfg_ready),
        .load_addr_valid  (load_addr_valid),
        .load_addr_ready  (load_addr_ready),
        .load_addr_data   (load_addr_data),
        .load_addr_tag    (load_addr_tag),
        .store_addr_valid (store_addr_valid),
        .store_addr_ready (store_addr_ready),
        .store_addr_data  (store_addr_data),
        .store_addr_tag   (store_addr_tag),
        .store_data_valid (store_data_valid),
        .store_data_ready (store_data_ready),
        .store_data_data  (store_data_data),
        .store_data_tag   (store_data_tag),
        .load_data_valid  (load_data_valid),
        .load_data_ready  (load_data_ready),
        .load_data_data   (load_data_data),
        .load_data_tag    (load_data_tag),
        .load_done_valid  (load_done_valid),
        .load_done_ready  (load_done_ready),
        .load_done_tag    (load_done_tag),
        .store_done_valid (store_done_valid),
        .store_done_ready (store_done_ready),
        .store_done_tag   (store_done_tag),
        // AXI tied off (IS_PRIVATE=1)
        .s_axi_awaddr     ('0),
        .s_axi_awvalid    (1'b0),
        .s_axi_awready    (),
        .s_axi_wdata      ('0),
        .s_axi_wstrb      ('0),
        .s_axi_wvalid     (1'b0),
        .s_axi_wready     (),
        .s_axi_bresp      (),
        .s_axi_bvalid     (),
        .s_axi_bready     (1'b0),
        .s_axi_araddr     ('0),
        .s_axi_arvalid    (1'b0),
        .s_axi_arready    (),
        .s_axi_rdata      (),
        .s_axi_rresp      (),
        .s_axi_rvalid     (),
        .s_axi_rready     (1'b0)
    );

    // Always accept load data, load done, store done
    assign load_data_ready  = 1'b1;
    assign load_done_ready  = 1'b1;
    assign store_done_ready = 1'b1;

    // =========================================================================
    // Test state machine
    // =========================================================================
    typedef enum logic [2:0] {
        ST_CFG_LOAD  = 3'd0,
        ST_WRITE     = 3'd1,
        ST_WRITE_DRAIN = 3'd2,
        ST_READ      = 3'd3,
        ST_READ_DRAIN = 3'd4,
        ST_VERDICT   = 3'd5
    } test_state_t;

    test_state_t test_state;

    // Config loading: 5 words for 1 region
    // Fields: valid=1, start_lane=0, end_lane=16, addr_offset=0, elem_size_log2=2
    integer cfg_idx;
    logic [31:0] cfg_table [0:4];

    initial begin : cfg_table_init
        cfg_table[0] = 32'd1;            // valid
        cfg_table[1] = 32'd0;            // start_lane
        cfg_table[2] = 32'd16;           // end_lane (covers tag 0..15)
        cfg_table[3] = 32'd0;            // addr_offset
        cfg_table[4] = 32'd2;            // elem_size_log2 = 2 (4 bytes)
    end

    // Write/read tracking
    integer write_addr_idx;
    integer write_data_idx;
    integer store_done_cnt;
    integer read_idx;
    integer read_done_cnt;
    integer recv_idx;
    integer mismatch_count;

    // Reference data
    logic [DATA_WIDTH-1:0] ref_data [0:NUM_WORDS-1];

    initial begin : ref_data_init
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < NUM_WORDS; iter_var0 = iter_var0 + 1) begin : init_loop
            ref_data[iter_var0] = (iter_var0 + 1) * 32'hDEAD0000 + iter_var0[DATA_WIDTH-1:0];
        end
    end

    // =========================================================================
    // Main sequential logic
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin : fsm_proc
        if (!rst_n) begin : fsm_reset
            test_state       <= ST_CFG_LOAD;
            cfg_idx          <= 0;
            cfg_valid        <= 1'b0;
            cfg_wdata        <= 32'd0;
            write_addr_idx   <= 0;
            write_data_idx   <= 0;
            store_done_cnt   <= 0;
            read_idx         <= 0;
            read_done_cnt    <= 0;
            recv_idx         <= 0;
            mismatch_count   <= 0;
            store_addr_valid <= 1'b0;
            store_addr_data  <= '0;
            store_addr_tag   <= '0;
            store_data_valid <= 1'b0;
            store_data_data  <= '0;
            store_data_tag   <= '0;
            load_addr_valid  <= 1'b0;
            load_addr_data   <= '0;
            load_addr_tag    <= '0;
        end else begin : fsm_active

            case (test_state)
                // ---------------------------------------------------------
                // Load region table config (5 words)
                // ---------------------------------------------------------
                ST_CFG_LOAD: begin : state_cfg_load
                    cfg_valid <= 1'b1;
                    cfg_wdata <= cfg_table[cfg_idx];
                    if (cfg_valid && cfg_ready) begin : cfg_advance
                        if (cfg_idx + 1 >= 5) begin : cfg_done
                            cfg_valid  <= 1'b0;
                            test_state <= ST_WRITE;
                        end else begin : cfg_next
                            cfg_idx <= cfg_idx + 1;
                        end
                    end
                end

                // ---------------------------------------------------------
                // Write phase: send store addr + store data for each word
                // ---------------------------------------------------------
                ST_WRITE: begin : state_write
                    cfg_valid <= 1'b0;

                    // Drive store address
                    if (write_addr_idx < NUM_WORDS) begin : write_addr_drive
                        store_addr_valid <= 1'b1;
                        store_addr_data  <= write_addr_idx[DATA_WIDTH-1:0];
                        store_addr_tag   <= '0;
                        if (store_addr_valid && store_addr_ready) begin : write_addr_advance
                            write_addr_idx <= write_addr_idx + 1;
                        end
                    end else begin : write_addr_done
                        store_addr_valid <= 1'b0;
                    end

                    // Drive store data
                    if (write_data_idx < NUM_WORDS) begin : write_data_drive
                        store_data_valid <= 1'b1;
                        store_data_data  <= ref_data[write_data_idx];
                        store_data_tag   <= '0;
                        if (store_data_valid && store_data_ready) begin : write_data_advance
                            write_data_idx <= write_data_idx + 1;
                        end
                    end else begin : write_data_done
                        store_data_valid <= 1'b0;
                    end

                    // Count store completions
                    if (store_done_valid && store_done_ready) begin : write_done_count
                        store_done_cnt <= store_done_cnt + 1;
                    end

                    // Transition when all writes issued and completed
                    if (write_addr_idx >= NUM_WORDS &&
                        write_data_idx >= NUM_WORDS) begin : write_to_drain
                        store_addr_valid <= 1'b0;
                        store_data_valid <= 1'b0;
                        test_state       <= ST_WRITE_DRAIN;
                    end
                end

                // ---------------------------------------------------------
                // Drain remaining store done signals
                // ---------------------------------------------------------
                ST_WRITE_DRAIN: begin : state_write_drain
                    if (store_done_valid && store_done_ready) begin : drain_done_count
                        store_done_cnt <= store_done_cnt + 1;
                    end
                    if (store_done_cnt >= NUM_WORDS) begin : drain_complete
                        test_state <= ST_READ;
                    end
                end

                // ---------------------------------------------------------
                // Read phase: issue load addresses
                // ---------------------------------------------------------
                ST_READ: begin : state_read
                    if (read_idx < NUM_WORDS) begin : read_drive
                        load_addr_valid <= 1'b1;
                        load_addr_data  <= read_idx[DATA_WIDTH-1:0];
                        load_addr_tag   <= '0;
                        if (load_addr_valid && load_addr_ready) begin : read_advance
                            read_idx <= read_idx + 1;
                        end
                    end else begin : read_done
                        load_addr_valid <= 1'b0;
                        test_state      <= ST_READ_DRAIN;
                    end

                    // Capture and verify returning load data
                    if (load_data_valid && load_data_ready) begin : read_check
                        if (load_data_data !== ref_data[recv_idx]) begin : read_mismatch
                            if (mismatch_count < 10) begin : read_report
                                $display("[tb_memory_sideeffect_test] MISMATCH[%0d]: got %h, expected %h",
                                         recv_idx, load_data_data, ref_data[recv_idx]);
                            end
                            mismatch_count <= mismatch_count + 1;
                        end
                        recv_idx <= recv_idx + 1;
                    end
                end

                // ---------------------------------------------------------
                // Drain remaining load responses
                // ---------------------------------------------------------
                ST_READ_DRAIN: begin : state_read_drain
                    if (load_data_valid && load_data_ready) begin : drain_read_check
                        if (load_data_data !== ref_data[recv_idx]) begin : drain_mismatch
                            if (mismatch_count < 10) begin : drain_report
                                $display("[tb_memory_sideeffect_test] MISMATCH[%0d]: got %h, expected %h",
                                         recv_idx, load_data_data, ref_data[recv_idx]);
                            end
                            mismatch_count <= mismatch_count + 1;
                        end
                        recv_idx <= recv_idx + 1;
                    end

                    if (recv_idx >= NUM_WORDS) begin : all_read
                        test_state <= ST_VERDICT;
                    end
                end

                // ---------------------------------------------------------
                // Verdict
                // ---------------------------------------------------------
                ST_VERDICT: begin : state_verdict
                    if (mismatch_count == 0) begin : verdict_pass
                        $display("[tb_memory_sideeffect_test] PASS: All %0d words written and read back correctly",
                                 NUM_WORDS);
                    end else begin : verdict_fail
                        $display("[tb_memory_sideeffect_test] FAIL: %0d mismatches out of %0d words",
                                 mismatch_count, NUM_WORDS);
                    end
                    $finish;
                end

                default: begin : state_default
                    test_state <= ST_VERDICT;
                end
            endcase
        end
    end

    // =========================================================================
    // Simulation timeout
    // =========================================================================
    integer cycle_count;

    always @(posedge clk or negedge rst_n) begin : timeout_proc
        if (!rst_n) begin : timeout_reset
            cycle_count <= 0;
        end else begin : timeout_active
            cycle_count <= cycle_count + 1;
            if (cycle_count >= SIM_TIMEOUT_CYCLES) begin : timeout_hit
                $display("[tb_memory_sideeffect_test] FAIL: Timeout at cycle %0d (state=%0d, wrote %0d, read %0d)",
                         cycle_count, test_state, write_addr_idx, recv_idx);
                $finish;
            end
        end
    end

endmodule
