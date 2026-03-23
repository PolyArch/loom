// tb_tapestry_dma.sv -- DMA engine test.
//
// Verifies SPM -> L2 transfer with data integrity checking.
// Uses a simplified L2 memory model to accept writes and return
// reads.
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

module tb_tapestry_dma;
  import mem_ctrl_pkg::*;

  // =========================================================================
  // Parameters
  // =========================================================================
  parameter DATA_WIDTH     = 32;
  parameter MAX_INFLIGHT   = 4;
  parameter SPM_SIZE_BYTES = 4096;
  parameter SPM_ADDR_WIDTH = 12;
  parameter XFER_BYTES     = 32;   // 8 words
  parameter CLK_PERIOD_NS  = 10;
  parameter RST_CYCLES     = 5;
  parameter TIMEOUT_CYCLES = 5000;

  localparam NUM_WORDS = XFER_BYTES / (DATA_WIDTH / 8);

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
  // SPM instance (used as source for DMA read)
  // =========================================================================
  logic [SPM_ADDR_WIDTH-1:0] core_addr;
  logic [DATA_WIDTH-1:0]     core_wdata;
  logic                      core_wr_en;
  logic                      core_req_valid;
  logic                      core_req_ready;
  logic [DATA_WIDTH-1:0]     core_rdata;
  logic                      core_rdata_valid;

  // DMA port signals (from DMA engine)
  logic [31:0]               dma_spm_addr;
  logic [31:0]               dma_spm_wdata;
  logic                      dma_spm_wr_en;
  logic                      dma_spm_req_valid;
  logic                      dma_spm_req_ready;
  logic [31:0]               dma_spm_rdata;
  logic                      dma_spm_rdata_valid;

  tapestry_spm #(
    .SPM_SIZE_BYTES (SPM_SIZE_BYTES),
    .DATA_WIDTH     (DATA_WIDTH),
    .ADDR_WIDTH     (SPM_ADDR_WIDTH)
  ) u_spm (
    .clk             (clk),
    .rst_n           (rst_n),
    .core_addr       (core_addr),
    .core_wdata      (core_wdata),
    .core_wr_en      (core_wr_en),
    .core_req_valid  (core_req_valid),
    .core_req_ready  (core_req_ready),
    .core_rdata      (core_rdata),
    .core_rdata_valid(core_rdata_valid),
    .dma_addr        (dma_spm_addr[SPM_ADDR_WIDTH-1:0]),
    .dma_wdata       (dma_spm_wdata[DATA_WIDTH-1:0]),
    .dma_wr_en       (dma_spm_wr_en),
    .dma_req_valid   (dma_spm_req_valid),
    .dma_req_ready   (dma_spm_req_ready),
    .dma_rdata       (dma_spm_rdata[DATA_WIDTH-1:0]),
    .dma_rdata_valid (dma_spm_rdata_valid)
  );

  // =========================================================================
  // DMA engine instance
  // =========================================================================
  dma_cmd_t  dma_cmd;
  logic      dma_cmd_valid;
  logic      dma_cmd_ready;
  logic      dma_busy;
  logic      dma_done;
  logic [15:0] dma_bytes_xferred;

  mem_req_t  ext_req;
  logic      ext_req_valid;
  logic      ext_req_ready;
  mem_resp_t ext_resp;
  logic      ext_resp_valid;
  logic      ext_resp_ready;

  tapestry_dma_engine #(
    .MAX_INFLIGHT (MAX_INFLIGHT),
    .DATA_WIDTH   (DATA_WIDTH)
  ) u_dma (
    .clk              (clk),
    .rst_n            (rst_n),
    .cmd              (dma_cmd),
    .cmd_valid        (dma_cmd_valid),
    .cmd_ready        (dma_cmd_ready),
    .busy             (dma_busy),
    .done             (dma_done),
    .bytes_transferred(dma_bytes_xferred),
    .spm_addr         (dma_spm_addr),
    .spm_wdata        (dma_spm_wdata),
    .spm_wr_en        (dma_spm_wr_en),
    .spm_req_valid    (dma_spm_req_valid),
    .spm_req_ready    (dma_spm_req_ready),
    .spm_rdata        (dma_spm_rdata),
    .spm_rdata_valid  (dma_spm_rdata_valid),
    .ext_req          (ext_req),
    .ext_req_valid    (ext_req_valid),
    .ext_req_ready    (ext_req_ready),
    .ext_resp         (ext_resp),
    .ext_resp_valid   (ext_resp_valid),
    .ext_resp_ready   (ext_resp_ready)
  );

  // =========================================================================
  // Simple L2 memory model (accepts writes, returns reads)
  // =========================================================================
  logic [DATA_WIDTH-1:0] l2_mem [0:1023];

  always_ff @(posedge clk or negedge rst_n) begin : l2_model
    if (!rst_n) begin : l2_model_reset
      ext_req_ready  <= 1'b1;
      ext_resp_valid <= 1'b0;
      ext_resp       <= '0;
    end : l2_model_reset
    else begin : l2_model_active
      ext_req_ready  <= 1'b1;
      ext_resp_valid <= 1'b0;

      if (ext_req_valid && ext_req_ready) begin : l2_accept
        if (ext_req.wr_en) begin : l2_write
          l2_mem[ext_req.addr[11:2]] <= ext_req.data;
          // Write response
          ext_resp_valid   <= 1'b1;
          ext_resp.data    <= '0;
          ext_resp.core_id <= ext_req.core_id;
          ext_resp.req_id  <= ext_req.req_id;
          ext_resp.error   <= 1'b0;
        end : l2_write
        else begin : l2_read
          ext_resp_valid   <= 1'b1;
          ext_resp.data    <= l2_mem[ext_req.addr[11:2]];
          ext_resp.core_id <= ext_req.core_id;
          ext_resp.req_id  <= ext_req.req_id;
          ext_resp.error   <= 1'b0;
        end : l2_read
      end : l2_accept
    end : l2_model_active
  end : l2_model

  // =========================================================================
  // Test state machine
  // =========================================================================
  typedef enum logic [2:0] {
    ST_FILL_SPM  = 3'd0,
    ST_START_DMA = 3'd1,
    ST_WAIT_DMA  = 3'd2,
    ST_VERIFY    = 3'd3,
    ST_VERDICT   = 3'd4
  } test_state_t;

  test_state_t test_state;
  integer word_idx;
  integer mismatch_count;

  // Reference data
  logic [DATA_WIDTH-1:0] ref_data [0:NUM_WORDS-1];

  initial begin : ref_init
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_WORDS; iter_var0 = iter_var0 + 1) begin : ref_loop
      ref_data[iter_var0] = 32'hDEAD0000 + iter_var0[DATA_WIDTH-1:0];
    end
  end

  // =========================================================================
  // Main FSM
  // =========================================================================
  always @(posedge clk or negedge rst_n) begin : fsm_proc
    if (!rst_n) begin : fsm_reset
      test_state     <= ST_FILL_SPM;
      word_idx       <= 0;
      mismatch_count <= 0;
      core_addr      <= '0;
      core_wdata     <= '0;
      core_wr_en     <= 1'b0;
      core_req_valid <= 1'b0;
      dma_cmd_valid  <= 1'b0;
      dma_cmd        <= '0;
    end : fsm_reset
    else begin : fsm_active
      case (test_state)

        // -- Fill SPM with test data via core port --
        ST_FILL_SPM: begin : state_fill_spm
          core_req_valid <= 1'b1;
          core_wr_en     <= 1'b1;
          core_addr      <= word_idx[SPM_ADDR_WIDTH-1:0] << 2;
          core_wdata     <= ref_data[word_idx];

          if (core_req_valid && core_req_ready) begin : fill_advance
            if (word_idx + 1 >= NUM_WORDS) begin : fill_done
              core_req_valid <= 1'b0;
              core_wr_en     <= 1'b0;
              word_idx       <= 0;
              test_state     <= ST_START_DMA;
            end : fill_done
            else begin : fill_next
              word_idx <= word_idx + 1;
            end : fill_next
          end : fill_advance
        end : state_fill_spm

        // -- Issue DMA command: SPM -> L2 --
        ST_START_DMA: begin : state_start_dma
          dma_cmd_valid        <= 1'b1;
          dma_cmd.src_addr     <= 32'h0000_0000;
          dma_cmd.dst_addr     <= 32'h0000_0000;
          dma_cmd.length       <= XFER_BYTES[15:0];
          dma_cmd.direction    <= 2'b00;  // SPM->L2
          dma_cmd.src_core_id  <= 4'd0;
          dma_cmd.dst_core_id  <= 4'd0;

          if (dma_cmd_valid && dma_cmd_ready) begin : dma_accepted
            dma_cmd_valid <= 1'b0;
            test_state    <= ST_WAIT_DMA;
          end : dma_accepted
        end : state_start_dma

        // -- Wait for DMA to complete --
        ST_WAIT_DMA: begin : state_wait_dma
          dma_cmd_valid <= 1'b0;
          if (dma_done) begin : dma_complete
            word_idx   <= 0;
            test_state <= ST_VERIFY;
          end : dma_complete
        end : state_wait_dma

        // -- Verify L2 memory contents --
        ST_VERIFY: begin : state_verify
          if (word_idx < NUM_WORDS) begin : verify_loop
            if (l2_mem[word_idx] !== ref_data[word_idx]) begin : verify_mismatch
              $display("[tb_tapestry_dma] L2 mismatch [%0d]: got=%h exp=%h",
                       word_idx, l2_mem[word_idx], ref_data[word_idx]);
              mismatch_count <= mismatch_count + 1;
            end : verify_mismatch
            word_idx <= word_idx + 1;
          end : verify_loop
          else begin : verify_done
            test_state <= ST_VERDICT;
          end : verify_done
        end : state_verify

        ST_VERDICT: begin : state_verdict
          if (mismatch_count == 0) begin : verdict_pass
            $display("[tb_tapestry_dma] PASS: DMA SPM->L2 transfer of %0d bytes correct",
                     XFER_BYTES);
          end : verdict_pass
          else begin : verdict_fail
            $display("[tb_tapestry_dma] FAIL: %0d mismatches in DMA transfer",
                     mismatch_count);
          end : verdict_fail
          $finish;
        end : state_verdict

        default: begin : state_default
          test_state <= ST_VERDICT;
        end : state_default
      endcase
    end : fsm_active
  end : fsm_proc

  // =========================================================================
  // Timeout
  // =========================================================================
  integer cycle_count;

  always @(posedge clk or negedge rst_n) begin : timeout_proc
    if (!rst_n) begin : timeout_reset
      cycle_count <= 0;
    end : timeout_reset
    else begin : timeout_active
      cycle_count <= cycle_count + 1;
      if (cycle_count >= TIMEOUT_CYCLES) begin : timeout_hit
        $display("[tb_tapestry_dma] FAIL: Timeout at cycle %0d (state=%0d, busy=%b, done=%b)",
                 cycle_count, test_state, dma_busy, dma_done);
        $finish;
      end : timeout_hit
    end : timeout_active
  end : timeout_proc

endmodule
