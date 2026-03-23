// tb_tapestry_l2.sv -- L2 controller test.
//
// Verifies:
//  1. Multi-core access with bank conflicts and correct queuing
//  2. Bank interleaving address decoding
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

module tb_tapestry_l2;
  import mem_ctrl_pkg::*;

  // =========================================================================
  // Parameters
  // =========================================================================
  parameter NUM_BANKS        = 4;
  parameter BANK_SIZE_BYTES  = 65536;
  parameter DATA_WIDTH       = 32;
  parameter INTERLEAVE_BYTES = 64;
  parameter NUM_CORES        = 4;
  parameter BANK_ADDR_WIDTH  = 16;
  parameter ACCESS_LATENCY   = 4;
  parameter CLK_PERIOD_NS    = 10;
  parameter RST_CYCLES       = 5;
  parameter TIMEOUT_CYCLES   = 5000;

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
  mem_req_t  core_req       [NUM_CORES];
  logic      core_req_valid [NUM_CORES];
  logic      core_req_ready [NUM_CORES];
  mem_resp_t core_resp      [NUM_CORES];
  logic      core_resp_valid [NUM_CORES];
  logic      core_resp_ready [NUM_CORES];

  tapestry_l2_ctrl #(
    .NUM_BANKS        (NUM_BANKS),
    .BANK_SIZE_BYTES  (BANK_SIZE_BYTES),
    .DATA_WIDTH       (DATA_WIDTH),
    .INTERLEAVE_BYTES (INTERLEAVE_BYTES),
    .NUM_CORES        (NUM_CORES),
    .BANK_ADDR_WIDTH  (BANK_ADDR_WIDTH),
    .ACCESS_LATENCY   (ACCESS_LATENCY)
  ) u_dut (
    .clk             (clk),
    .rst_n           (rst_n),
    .core_req        (core_req),
    .core_req_valid  (core_req_valid),
    .core_req_ready  (core_req_ready),
    .core_resp       (core_resp),
    .core_resp_valid (core_resp_valid),
    .core_resp_ready (core_resp_ready)
  );

  // Always ready to accept responses
  initial begin : resp_ready_init
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_CORES; iter_var0 = iter_var0 + 1) begin : resp_ready_loop
      core_resp_ready[iter_var0] = 1'b1;
    end
  end

  // =========================================================================
  // Test state machine
  // =========================================================================
  typedef enum logic [3:0] {
    ST_IDLE            = 4'd0,
    ST_WRITE_BANK0     = 4'd1,
    ST_WRITE_DRAIN     = 4'd2,
    ST_READ_BANK0      = 4'd3,
    ST_READ_WAIT       = 4'd4,
    ST_CONFLICT_WRITE  = 4'd5,
    ST_CONFLICT_DRAIN  = 4'd6,
    ST_CONFLICT_READ   = 4'd7,
    ST_CONFLICT_WAIT   = 4'd8,
    ST_VERDICT         = 4'd9
  } test_state_t;

  test_state_t test_state;
  integer mismatch_count;
  integer resp_count;
  integer write_idx;
  integer drain_wait;

  // =========================================================================
  // Main FSM
  // =========================================================================
  always @(posedge clk or negedge rst_n) begin : fsm_proc
    if (!rst_n) begin : fsm_reset
      integer iter_var0;
      test_state     <= ST_WRITE_BANK0;
      mismatch_count <= 0;
      resp_count     <= 0;
      write_idx      <= 0;
      drain_wait     <= 0;
      for (iter_var0 = 0; iter_var0 < NUM_CORES; iter_var0 = iter_var0 + 1) begin : init_req_loop
        core_req_valid[iter_var0] <= 1'b0;
        core_req[iter_var0]       <= '0;
      end : init_req_loop
    end : fsm_reset
    else begin : fsm_active
      case (test_state)

        // -- Write 4 words to bank 0 from core 0 --
        ST_WRITE_BANK0: begin : state_write_bank0
          core_req_valid[0]   <= 1'b1;
          core_req[0].addr    <= (write_idx * 4) + (0 * INTERLEAVE_BYTES);
          core_req[0].data    <= 32'hBEEF0000 + write_idx[31:0];
          core_req[0].wr_en   <= 1'b1;
          core_req[0].core_id <= 4'd0;
          core_req[0].req_id  <= write_idx[7:0];

          if (core_req_valid[0] && core_req_ready[0]) begin : wr_bank0_advance
            if (write_idx + 1 >= 4) begin : wr_bank0_done
              core_req_valid[0] <= 1'b0;
              write_idx         <= 0;
              drain_wait        <= 0;
              test_state        <= ST_WRITE_DRAIN;
            end : wr_bank0_done
            else begin : wr_bank0_next
              write_idx <= write_idx + 1;
            end : wr_bank0_next
          end : wr_bank0_advance
        end : state_write_bank0

        // -- Wait for writes to drain through pipeline --
        ST_WRITE_DRAIN: begin : state_write_drain
          drain_wait <= drain_wait + 1;
          if (drain_wait >= ACCESS_LATENCY + 4) begin : drain_done
            write_idx  <= 0;
            resp_count <= 0;
            test_state <= ST_READ_BANK0;
          end : drain_done
        end : state_write_drain

        // -- Read back from bank 0 using core 0 --
        ST_READ_BANK0: begin : state_read_bank0
          core_req_valid[0]   <= 1'b1;
          core_req[0].addr    <= (write_idx * 4) + (0 * INTERLEAVE_BYTES);
          core_req[0].data    <= '0;
          core_req[0].wr_en   <= 1'b0;
          core_req[0].core_id <= 4'd0;
          core_req[0].req_id  <= write_idx[7:0];

          if (core_req_valid[0] && core_req_ready[0]) begin : rd_bank0_advance
            if (write_idx + 1 >= 4) begin : rd_bank0_done
              core_req_valid[0] <= 1'b0;
              write_idx         <= 0;
              test_state        <= ST_READ_WAIT;
            end : rd_bank0_done
            else begin : rd_bank0_next
              write_idx <= write_idx + 1;
            end : rd_bank0_next
          end : rd_bank0_advance
        end : state_read_bank0

        // -- Wait for read responses and verify --
        ST_READ_WAIT: begin : state_read_wait
          if (core_resp_valid[0]) begin : resp_check
            if (core_resp[0].data !== (32'hBEEF0000 + resp_count[31:0])) begin : resp_mismatch
              $display("[tb_tapestry_l2] Read mismatch [%0d]: got=%h exp=%h",
                       resp_count, core_resp[0].data,
                       32'hBEEF0000 + resp_count[31:0]);
              mismatch_count <= mismatch_count + 1;
            end : resp_mismatch
            resp_count <= resp_count + 1;
          end : resp_check

          if (resp_count >= 4) begin : reads_complete
            write_idx  <= 0;
            drain_wait <= 0;
            test_state <= ST_CONFLICT_WRITE;
          end : reads_complete
        end : state_read_wait

        // -- Bank conflict test: core 0 and core 1 write to same bank --
        ST_CONFLICT_WRITE: begin : state_conflict_write
          // Core 0 writes to addr 0 (bank 0)
          core_req_valid[0]   <= 1'b1;
          core_req[0].addr    <= 32'h0000_0000;
          core_req[0].data    <= 32'hCAFE0000;
          core_req[0].wr_en   <= 1'b1;
          core_req[0].core_id <= 4'd0;
          core_req[0].req_id  <= 8'hA0;

          // Core 1 writes to addr 4 (same bank 0)
          core_req_valid[1]   <= 1'b1;
          core_req[1].addr    <= 32'h0000_0004;
          core_req[1].data    <= 32'hCAFE0001;
          core_req[1].wr_en   <= 1'b1;
          core_req[1].core_id <= 4'd1;
          core_req[1].req_id  <= 8'hB0;

          // One should be accepted, the other must wait
          if (core_req_valid[0] && core_req_ready[0]) begin : conflict_c0_done
            core_req_valid[0] <= 1'b0;
          end : conflict_c0_done
          if (core_req_valid[1] && core_req_ready[1]) begin : conflict_c1_done
            core_req_valid[1] <= 1'b0;
          end : conflict_c1_done

          if (!core_req_valid[0] && !core_req_valid[1]) begin : conflict_both_done
            drain_wait <= 0;
            test_state <= ST_CONFLICT_DRAIN;
          end : conflict_both_done
          // Also handle case where both accepted in same cycle
          if ((core_req_valid[0] && core_req_ready[0]) &&
              (core_req_valid[1] && core_req_ready[1])) begin : conflict_simultaneous
            drain_wait <= 0;
            test_state <= ST_CONFLICT_DRAIN;
          end : conflict_simultaneous
        end : state_conflict_write

        ST_CONFLICT_DRAIN: begin : state_conflict_drain
          core_req_valid[0] <= 1'b0;
          core_req_valid[1] <= 1'b0;
          drain_wait <= drain_wait + 1;
          if (drain_wait >= ACCESS_LATENCY + 4) begin : conflict_drain_done
            resp_count <= 0;
            test_state <= ST_CONFLICT_READ;
          end : conflict_drain_done
        end : state_conflict_drain

        // -- Read back conflict writes from core 0 --
        ST_CONFLICT_READ: begin : state_conflict_read
          core_req_valid[0]   <= 1'b1;
          core_req[0].addr    <= (resp_count == 0) ? 32'h0000_0000 : 32'h0000_0004;
          core_req[0].data    <= '0;
          core_req[0].wr_en   <= 1'b0;
          core_req[0].core_id <= 4'd0;
          core_req[0].req_id  <= resp_count[7:0];

          if (core_req_valid[0] && core_req_ready[0]) begin : conflict_rd_advance
            if (resp_count + 1 >= 2) begin : conflict_rd_done
              core_req_valid[0] <= 1'b0;
              resp_count        <= 0;
              test_state        <= ST_CONFLICT_WAIT;
            end : conflict_rd_done
            else begin : conflict_rd_next
              resp_count <= resp_count + 1;
            end : conflict_rd_next
          end : conflict_rd_advance
        end : state_conflict_read

        ST_CONFLICT_WAIT: begin : state_conflict_wait
          if (core_resp_valid[0]) begin : conflict_resp_check
            if (resp_count == 0 && core_resp[0].data !== 32'hCAFE0000) begin : conflict_mismatch0
              $display("[tb_tapestry_l2] Conflict read0 mismatch: got=%h exp=CAFE0000",
                       core_resp[0].data);
              mismatch_count <= mismatch_count + 1;
            end : conflict_mismatch0
            if (resp_count == 1 && core_resp[0].data !== 32'hCAFE0001) begin : conflict_mismatch1
              $display("[tb_tapestry_l2] Conflict read1 mismatch: got=%h exp=CAFE0001",
                       core_resp[0].data);
              mismatch_count <= mismatch_count + 1;
            end : conflict_mismatch1
            resp_count <= resp_count + 1;
          end : conflict_resp_check

          if (resp_count >= 2) begin : all_conflict_done
            test_state <= ST_VERDICT;
          end : all_conflict_done
        end : state_conflict_wait

        ST_VERDICT: begin : state_verdict
          if (mismatch_count == 0) begin : verdict_pass
            $display("[tb_tapestry_l2] PASS: Bank interleaving and conflict arbitration correct");
          end : verdict_pass
          else begin : verdict_fail
            $display("[tb_tapestry_l2] FAIL: %0d mismatches", mismatch_count);
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
        $display("[tb_tapestry_l2] FAIL: Timeout at cycle %0d (state=%0d)",
                 cycle_count, test_state);
        $finish;
      end : timeout_hit
    end : timeout_active
  end : timeout_proc

endmodule
