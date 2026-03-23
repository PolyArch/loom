// tapestry_dma_engine.sv -- DMA engine per core.
//
// Command-driven DMA engine that transfers data between SPM and
// L2/DRAM.  Supports a configurable number of in-flight words.
//
// State machine: IDLE -> SETUP -> TRANSFER -> DONE
//
// During TRANSFER, the engine generates burst read/write requests
// and tracks progress via bytes_transferred.

module tapestry_dma_engine
  import mem_ctrl_pkg::*;
#(
  parameter int unsigned MAX_INFLIGHT = 4,
  parameter int unsigned DATA_WIDTH   = 32
)(
  input  logic          clk,
  input  logic          rst_n,

  // --- Command interface (from core config registers) ---
  input  dma_cmd_t      cmd,
  input  logic          cmd_valid,
  output logic          cmd_ready,

  // --- Status ---
  output logic          busy,
  output logic          done,
  output logic [15:0]   bytes_transferred,

  // --- SPM interface ---
  output logic [31:0]   spm_addr,
  output logic [31:0]   spm_wdata,
  output logic          spm_wr_en,
  output logic          spm_req_valid,
  input  logic          spm_req_ready,
  input  logic [31:0]   spm_rdata,
  input  logic          spm_rdata_valid,

  // --- L2/DRAM interface (via NoC) ---
  output mem_req_t      ext_req,
  output logic          ext_req_valid,
  input  logic          ext_req_ready,
  /* verilator lint_off UNUSEDSIGNAL */
  input  mem_resp_t     ext_resp,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic          ext_resp_valid,
  output logic          ext_resp_ready
);

  // ---------------------------------------------------------------
  // Constants
  // ---------------------------------------------------------------
  localparam int unsigned BYTES_PER_WORD = DATA_WIDTH / 8;

  // ---------------------------------------------------------------
  // State encoding
  // ---------------------------------------------------------------
  typedef enum logic [2:0] {
    DMA_IDLE     = 3'd0,
    DMA_SETUP    = 3'd1,
    DMA_RD_SRC   = 3'd2,
    DMA_WR_DST   = 3'd3,
    DMA_DONE     = 3'd4
  } dma_state_t;

  dma_state_t state;

  // ---------------------------------------------------------------
  // Command register
  // ---------------------------------------------------------------
  /* verilator lint_off UNUSEDSIGNAL */
  dma_cmd_t cmd_reg;
  /* verilator lint_on UNUSEDSIGNAL */

  // Transfer counters
  logic [15:0] rd_offset;     // byte offset for reads
  logic [15:0] wr_offset;     // byte offset for writes
  logic [15:0] total_bytes;
  logic [15:0] bytes_xferred;

  // Data buffer for holding words read from source
  logic [DATA_WIDTH-1:0] data_buf [0:MAX_INFLIGHT-1];
  logic [$clog2(MAX_INFLIGHT+1)-1:0] buf_wr_ptr;
  logic [$clog2(MAX_INFLIGHT+1)-1:0] buf_rd_ptr;
  logic [$clog2(MAX_INFLIGHT+1)-1:0] buf_count;

  // Direction decode
  logic dir_spm_to_ext;  // SPM->L2 or SPM->DRAM
  logic dir_ext_to_spm;  // L2->SPM or DRAM->SPM

  assign dir_spm_to_ext = (cmd_reg.direction == 2'b00) |
                           (cmd_reg.direction == 2'b10);
  assign dir_ext_to_spm = (cmd_reg.direction == 2'b01) |
                           (cmd_reg.direction == 2'b11);

  // ---------------------------------------------------------------
  // Outputs
  // ---------------------------------------------------------------
  assign busy = (state != DMA_IDLE);
  assign done = (state == DMA_DONE);
  assign bytes_transferred = bytes_xferred;

  // Command is accepted only in IDLE state
  assign cmd_ready = (state == DMA_IDLE);

  // ---------------------------------------------------------------
  // Request ID tracking
  // ---------------------------------------------------------------
  logic [7:0] req_id_cnt;

  // ---------------------------------------------------------------
  // Main state machine
  // ---------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin : dma_fsm
    if (!rst_n) begin : dma_fsm_reset
      state        <= DMA_IDLE;
      cmd_reg      <= '0;
      rd_offset    <= '0;
      wr_offset    <= '0;
      total_bytes  <= '0;
      bytes_xferred <= '0;
      buf_wr_ptr   <= '0;
      buf_rd_ptr   <= '0;
      buf_count    <= '0;
      req_id_cnt   <= '0;
    end : dma_fsm_reset
    else begin : dma_fsm_active
      case (state)
        // -----------------------------------------------------------
        // IDLE: wait for command
        // -----------------------------------------------------------
        DMA_IDLE: begin : idle_state
          if (cmd_valid) begin : idle_accept
            cmd_reg      <= cmd;
            total_bytes  <= cmd.length;
            rd_offset    <= '0;
            wr_offset    <= '0;
            bytes_xferred <= '0;
            buf_wr_ptr   <= '0;
            buf_rd_ptr   <= '0;
            buf_count    <= '0;
            req_id_cnt   <= '0;
            state        <= DMA_SETUP;
          end : idle_accept
        end : idle_state

        // -----------------------------------------------------------
        // SETUP: prepare for transfer
        // -----------------------------------------------------------
        DMA_SETUP: begin : setup_state
          state <= DMA_RD_SRC;
        end : setup_state

        // -----------------------------------------------------------
        // RD_SRC: read words from source
        // -----------------------------------------------------------
        DMA_RD_SRC: begin : rd_src_state
          // Fill the buffer by reading from source
          if (dir_spm_to_ext) begin : rd_from_spm
            // Reading from SPM
            if (spm_req_valid && spm_req_ready) begin : spm_rd_accept
              rd_offset <= rd_offset + BYTES_PER_WORD[15:0];
            end : spm_rd_accept
            // Capture SPM read data into buffer
            if (spm_rdata_valid && buf_count < MAX_INFLIGHT[$clog2(MAX_INFLIGHT+1)-1:0]) begin : spm_rd_capture
              data_buf[buf_wr_ptr[$clog2(MAX_INFLIGHT)-1:0]] <= spm_rdata;
              buf_wr_ptr <= buf_wr_ptr + 1'b1;
              buf_count  <= buf_count + 1'b1;
            end : spm_rd_capture
          end : rd_from_spm
          else begin : rd_from_ext
            // Reading from L2/DRAM
            if (ext_req_valid && ext_req_ready) begin : ext_rd_accept
              rd_offset  <= rd_offset + BYTES_PER_WORD[15:0];
              req_id_cnt <= req_id_cnt + 1'b1;
            end : ext_rd_accept
            // Capture external read response into buffer
            if (ext_resp_valid && buf_count < MAX_INFLIGHT[$clog2(MAX_INFLIGHT+1)-1:0]) begin : ext_rd_capture
              data_buf[buf_wr_ptr[$clog2(MAX_INFLIGHT)-1:0]] <= ext_resp.data;
              buf_wr_ptr <= buf_wr_ptr + 1'b1;
              buf_count  <= buf_count + 1'b1;
            end : ext_rd_capture
          end : rd_from_ext

          // Transition to write phase when buffer has data
          if (buf_count > 0 || rd_offset >= total_bytes) begin : rd_to_wr
            state <= DMA_WR_DST;
          end : rd_to_wr
        end : rd_src_state

        // -----------------------------------------------------------
        // WR_DST: write buffered words to destination
        // -----------------------------------------------------------
        DMA_WR_DST: begin : wr_dst_state
          if (dir_spm_to_ext) begin : wr_to_ext
            // Writing to L2/DRAM
            if (ext_req_valid && ext_req_ready && buf_count > 0) begin : ext_wr_accept
              wr_offset    <= wr_offset + BYTES_PER_WORD[15:0];
              bytes_xferred <= bytes_xferred + BYTES_PER_WORD[15:0];
              buf_rd_ptr   <= buf_rd_ptr + 1'b1;
              buf_count    <= buf_count - 1'b1;
              req_id_cnt   <= req_id_cnt + 1'b1;
            end : ext_wr_accept
          end : wr_to_ext
          else begin : wr_to_spm
            // Writing to SPM
            if (spm_req_valid && spm_req_ready && buf_count > 0) begin : spm_wr_accept
              wr_offset    <= wr_offset + BYTES_PER_WORD[15:0];
              bytes_xferred <= bytes_xferred + BYTES_PER_WORD[15:0];
              buf_rd_ptr   <= buf_rd_ptr + 1'b1;
              buf_count    <= buf_count - 1'b1;
            end : spm_wr_accept
          end : wr_to_spm

          // Switch back to read more data or finish
          if (buf_count == 0 || (buf_count == 1 &&
              ((dir_spm_to_ext && ext_req_valid && ext_req_ready) ||
               (!dir_spm_to_ext && spm_req_valid && spm_req_ready)))) begin : wr_check_done
            if (wr_offset + BYTES_PER_WORD[15:0] >= total_bytes &&
                ((dir_spm_to_ext && ext_req_valid && ext_req_ready) ||
                 (!dir_spm_to_ext && spm_req_valid && spm_req_ready) ||
                 wr_offset >= total_bytes)) begin : wr_complete
              state <= DMA_DONE;
            end : wr_complete
            else if (rd_offset < total_bytes) begin : wr_more_read
              state <= DMA_RD_SRC;
            end : wr_more_read
          end : wr_check_done
        end : wr_dst_state

        // -----------------------------------------------------------
        // DONE: transfer complete, return to idle
        // -----------------------------------------------------------
        DMA_DONE: begin : done_state
          state <= DMA_IDLE;
        end : done_state

        default: begin : default_state
          state <= DMA_IDLE;
        end : default_state
      endcase
    end : dma_fsm_active
  end : dma_fsm

  // ---------------------------------------------------------------
  // SPM interface drive
  // ---------------------------------------------------------------
  always_comb begin : spm_drive
    spm_addr      = '0;
    spm_wdata     = '0;
    spm_wr_en     = 1'b0;
    spm_req_valid = 1'b0;

    if (state == DMA_RD_SRC && dir_spm_to_ext) begin : spm_rd_drive
      // Read from SPM
      spm_addr      = cmd_reg.src_addr + {16'b0, rd_offset};
      spm_wr_en     = 1'b0;
      spm_req_valid = (rd_offset < total_bytes) &
                      (buf_count < MAX_INFLIGHT[$clog2(MAX_INFLIGHT+1)-1:0]);
    end : spm_rd_drive
    else if (state == DMA_WR_DST && dir_ext_to_spm) begin : spm_wr_drive
      // Write to SPM
      spm_addr      = cmd_reg.dst_addr + {16'b0, wr_offset};
      spm_wdata     = data_buf[buf_rd_ptr[$clog2(MAX_INFLIGHT)-1:0]];
      spm_wr_en     = 1'b1;
      spm_req_valid = (buf_count > 0);
    end : spm_wr_drive
  end : spm_drive

  // ---------------------------------------------------------------
  // External (L2/DRAM) interface drive
  // ---------------------------------------------------------------
  always_comb begin : ext_drive
    ext_req         = '0;
    ext_req_valid   = 1'b0;
    ext_resp_ready  = 1'b0;

    if (state == DMA_RD_SRC && dir_ext_to_spm) begin : ext_rd_drive
      // Read from L2/DRAM
      ext_req.addr    = cmd_reg.src_addr + {16'b0, rd_offset};
      ext_req.data    = '0;
      ext_req.wr_en   = 1'b0;
      ext_req.core_id = cmd_reg.src_core_id;
      ext_req.req_id  = req_id_cnt;
      ext_req_valid   = (rd_offset < total_bytes) &
                        (buf_count < MAX_INFLIGHT[$clog2(MAX_INFLIGHT+1)-1:0]);
      ext_resp_ready  = (buf_count < MAX_INFLIGHT[$clog2(MAX_INFLIGHT+1)-1:0]);
    end : ext_rd_drive
    else if (state == DMA_WR_DST && dir_spm_to_ext) begin : ext_wr_drive
      // Write to L2/DRAM
      ext_req.addr    = cmd_reg.dst_addr + {16'b0, wr_offset};
      ext_req.data    = data_buf[buf_rd_ptr[$clog2(MAX_INFLIGHT)-1:0]];
      ext_req.wr_en   = 1'b1;
      ext_req.core_id = cmd_reg.dst_core_id;
      ext_req.req_id  = req_id_cnt;
      ext_req_valid   = (buf_count > 0);
      ext_resp_ready  = 1'b1;
    end : ext_wr_drive
  end : ext_drive

endmodule : tapestry_dma_engine
