// tb_tapestry_spm.sv -- SPM unit test.
//
// Verifies:
//  1. Read-after-write returns correct data
//  2. Simultaneous core and DMA access (core has priority)
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

module tb_tapestry_spm;

  // =========================================================================
  // Parameters
  // =========================================================================
  parameter SPM_SIZE_BYTES = 4096;
  parameter DATA_WIDTH     = 32;
  parameter ADDR_WIDTH     = 12;
  parameter NUM_WORDS      = 8;
  parameter CLK_PERIOD_NS  = 10;
  parameter RST_CYCLES     = 5;
  parameter TIMEOUT_CYCLES = 2000;

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
  // DUT signals
  // =========================================================================
  logic [ADDR_WIDTH-1:0]  core_addr;
  logic [DATA_WIDTH-1:0]  core_wdata;
  logic                   core_wr_en;
  logic                   core_req_valid;
  logic                   core_req_ready;
  logic [DATA_WIDTH-1:0]  core_rdata;
  logic                   core_rdata_valid;

  logic [ADDR_WIDTH-1:0]  dma_addr;
  logic [DATA_WIDTH-1:0]  dma_wdata;
  logic                   dma_wr_en;
  logic                   dma_req_valid;
  logic                   dma_req_ready;
  logic [DATA_WIDTH-1:0]  dma_rdata;
  logic                   dma_rdata_valid;

  tapestry_spm #(
    .SPM_SIZE_BYTES (SPM_SIZE_BYTES),
    .DATA_WIDTH     (DATA_WIDTH),
    .ADDR_WIDTH     (ADDR_WIDTH)
  ) u_dut (
    .clk             (clk),
    .rst_n           (rst_n),
    .core_addr       (core_addr),
    .core_wdata      (core_wdata),
    .core_wr_en      (core_wr_en),
    .core_req_valid  (core_req_valid),
    .core_req_ready  (core_req_ready),
    .core_rdata      (core_rdata),
    .core_rdata_valid(core_rdata_valid),
    .dma_addr        (dma_addr),
    .dma_wdata       (dma_wdata),
    .dma_wr_en       (dma_wr_en),
    .dma_req_valid   (dma_req_valid),
    .dma_req_ready   (dma_req_ready),
    .dma_rdata       (dma_rdata),
    .dma_rdata_valid (dma_rdata_valid)
  );

  // =========================================================================
  // Test state machine
  // =========================================================================
  typedef enum logic [3:0] {
    ST_IDLE         = 4'd0,
    ST_CORE_WRITE   = 4'd1,
    ST_CORE_READ    = 4'd2,
    ST_CORE_VERIFY  = 4'd3,
    ST_CONFLICT_WR  = 4'd4,
    ST_CONFLICT_CHK = 4'd5,
    ST_DMA_READ     = 4'd6,
    ST_DMA_VERIFY   = 4'd7,
    ST_VERDICT      = 4'd8
  } test_state_t;

  test_state_t test_state;
  integer word_idx;
  integer mismatch_count;
  integer dma_blocked_count;

  // Reference data
  logic [DATA_WIDTH-1:0] ref_data [0:NUM_WORDS-1];

  initial begin : ref_init
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_WORDS; iter_var0 = iter_var0 + 1) begin : ref_loop
      ref_data[iter_var0] = 32'hA5A50000 + iter_var0[DATA_WIDTH-1:0];
    end
  end

  // =========================================================================
  // Main FSM
  // =========================================================================
  always @(posedge clk or negedge rst_n) begin : fsm_proc
    if (!rst_n) begin : fsm_reset
      test_state       <= ST_CORE_WRITE;
      word_idx         <= 0;
      mismatch_count   <= 0;
      dma_blocked_count <= 0;
      core_addr        <= '0;
      core_wdata       <= '0;
      core_wr_en       <= 1'b0;
      core_req_valid   <= 1'b0;
      dma_addr         <= '0;
      dma_wdata        <= '0;
      dma_wr_en        <= 1'b0;
      dma_req_valid    <= 1'b0;
    end : fsm_reset
    else begin : fsm_active
      case (test_state)

        // -- Write NUM_WORDS via core port --
        ST_CORE_WRITE: begin : state_core_write
          core_req_valid <= 1'b1;
          core_wr_en     <= 1'b1;
          core_addr      <= word_idx[ADDR_WIDTH-1:0] << 2;
          core_wdata     <= ref_data[word_idx];
          dma_req_valid  <= 1'b0;

          if (core_req_valid && core_req_ready) begin : core_wr_advance
            if (word_idx + 1 >= NUM_WORDS) begin : core_wr_done
              core_req_valid <= 1'b0;
              core_wr_en     <= 1'b0;
              word_idx       <= 0;
              test_state     <= ST_CORE_READ;
            end : core_wr_done
            else begin : core_wr_next
              word_idx <= word_idx + 1;
            end : core_wr_next
          end : core_wr_advance
        end : state_core_write

        // -- Read back via core port --
        ST_CORE_READ: begin : state_core_read
          core_req_valid <= 1'b1;
          core_wr_en     <= 1'b0;
          core_addr      <= word_idx[ADDR_WIDTH-1:0] << 2;

          if (core_req_valid && core_req_ready) begin : core_rd_advance
            if (word_idx + 1 >= NUM_WORDS) begin : core_rd_done
              core_req_valid <= 1'b0;
              word_idx       <= 0;
              test_state     <= ST_CORE_VERIFY;
            end : core_rd_done
            else begin : core_rd_next
              word_idx <= word_idx + 1;
            end : core_rd_next
          end : core_rd_advance
        end : state_core_read

        // -- Verify core reads --
        ST_CORE_VERIFY: begin : state_core_verify
          if (core_rdata_valid) begin : verify_check
            if (core_rdata !== ref_data[word_idx]) begin : verify_mismatch
              $display("[tb_tapestry_spm] Core read mismatch [%0d]: got=%h exp=%h",
                       word_idx, core_rdata, ref_data[word_idx]);
              mismatch_count <= mismatch_count + 1;
            end : verify_mismatch
            if (word_idx + 1 >= NUM_WORDS) begin : verify_done
              word_idx   <= 0;
              test_state <= ST_CONFLICT_WR;
            end : verify_done
            else begin : verify_next
              word_idx <= word_idx + 1;
            end : verify_next
          end : verify_check
        end : state_core_verify

        // -- Conflict test: core and DMA request simultaneously --
        ST_CONFLICT_WR: begin : state_conflict_wr
          core_req_valid <= 1'b1;
          core_wr_en     <= 1'b0;
          core_addr      <= 12'h000;
          dma_req_valid  <= 1'b1;
          dma_wr_en      <= 1'b0;
          dma_addr       <= 12'h004;

          // DMA should be blocked (dma_req_ready=0 while core is active)
          if (!dma_req_ready && dma_req_valid) begin : conflict_blocked
            dma_blocked_count <= dma_blocked_count + 1;
          end : conflict_blocked

          // After one cycle, deassert core to let DMA through
          if (dma_blocked_count > 0) begin : conflict_release
            core_req_valid <= 1'b0;
            test_state     <= ST_CONFLICT_CHK;
          end : conflict_release
        end : state_conflict_wr

        // -- Verify DMA gets access after core releases --
        ST_CONFLICT_CHK: begin : state_conflict_chk
          core_req_valid <= 1'b0;
          dma_req_valid  <= 1'b1;
          dma_wr_en      <= 1'b0;
          dma_addr       <= 12'h000;

          if (dma_req_ready) begin : dma_accepted
            dma_req_valid <= 1'b0;
            test_state    <= ST_DMA_VERIFY;
          end : dma_accepted
        end : state_conflict_chk

        // -- DMA verify: check that DMA read returns correct data --
        ST_DMA_VERIFY: begin : state_dma_verify
          dma_req_valid <= 1'b0;
          if (dma_rdata_valid) begin : dma_check
            if (dma_rdata !== ref_data[0]) begin : dma_mismatch
              $display("[tb_tapestry_spm] DMA read mismatch: got=%h exp=%h",
                       dma_rdata, ref_data[0]);
              mismatch_count <= mismatch_count + 1;
            end : dma_mismatch
            test_state <= ST_VERDICT;
          end : dma_check
        end : state_dma_verify

        // -- Verdict --
        ST_VERDICT: begin : state_verdict
          if (mismatch_count == 0 && dma_blocked_count > 0) begin : verdict_pass
            $display("[tb_tapestry_spm] PASS: Core R/W correct, DMA priority correct (blocked %0d cycles)",
                     dma_blocked_count);
          end : verdict_pass
          else begin : verdict_fail
            $display("[tb_tapestry_spm] FAIL: mismatches=%0d, dma_blocked=%0d",
                     mismatch_count, dma_blocked_count);
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
        $display("[tb_tapestry_spm] FAIL: Timeout at cycle %0d (state=%0d)",
                 cycle_count, test_state);
        $finish;
      end : timeout_hit
    end : timeout_active
  end : timeout_proc

endmodule
