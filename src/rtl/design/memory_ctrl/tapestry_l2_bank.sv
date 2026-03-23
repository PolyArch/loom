// tapestry_l2_bank.sv -- Single L2 cache bank.
//
// Provides a single-port SRAM bank with valid-ready request/response
// interface.  Access latency is configurable via ACCESS_LATENCY
// parameter (pipeline register stages).
//
// The SRAM is behaviorally described for synthesis mapping.

module tapestry_l2_bank
  import mem_ctrl_pkg::*;
#(
  parameter int unsigned BANK_SIZE_BYTES = 65536,  // 64KB per bank
  parameter int unsigned DATA_WIDTH      = 32,
  /* verilator lint_off UNUSEDPARAM */
  parameter int unsigned ADDR_WIDTH      = 16,     // log2(BANK_SIZE_BYTES)
  /* verilator lint_on UNUSEDPARAM */
  parameter int unsigned ACCESS_LATENCY  = 4       // cycles of pipeline delay
)(
  input  logic        clk,
  input  logic        rst_n,

  // --- Request port (from arbiter) ---
  input  mem_req_t    req,
  input  logic        req_valid,
  output logic        req_ready,

  // --- Response port ---
  output mem_resp_t   resp,
  output logic        resp_valid,
  input  logic        resp_ready
);

  // ---------------------------------------------------------------
  // Localparams
  // ---------------------------------------------------------------
  localparam int unsigned NUM_WORDS = BANK_SIZE_BYTES / (DATA_WIDTH / 8);
  localparam int unsigned WORD_AW   = $clog2(NUM_WORDS > 1 ? NUM_WORDS : 2);

  // Pipeline depth: at least 1
  localparam int unsigned PIPE_DEPTH = (ACCESS_LATENCY > 0) ? ACCESS_LATENCY : 1;

  // ---------------------------------------------------------------
  // Storage array
  // ---------------------------------------------------------------
  (* ram_style = "auto" *)
  logic [DATA_WIDTH-1:0] mem [0:NUM_WORDS-1];

  // ---------------------------------------------------------------
  // Internal response buffer (to handle resp_ready back-pressure)
  // ---------------------------------------------------------------
  logic resp_buf_valid;
  mem_resp_t resp_buf;

  // Accept a new request when we can push into pipeline and no
  // response is stalled.
  assign req_ready = ~resp_buf_valid | resp_ready;

  // ---------------------------------------------------------------
  // Word address extraction from byte address
  // ---------------------------------------------------------------
  logic [WORD_AW-1:0] word_addr;
  assign word_addr = req.addr[WORD_AW+1:2]; // word-aligned addressing

  // ---------------------------------------------------------------
  // Pipeline shift register for tracking in-flight reads
  // ---------------------------------------------------------------
  logic                  pipe_valid [0:PIPE_DEPTH-1];
  logic [3:0]            pipe_core_id [0:PIPE_DEPTH-1];
  logic [7:0]            pipe_req_id  [0:PIPE_DEPTH-1];
  logic                  pipe_is_read [0:PIPE_DEPTH-1];
  logic [DATA_WIDTH-1:0] pipe_rdata   [0:PIPE_DEPTH-1];

  // ---------------------------------------------------------------
  // SRAM write (on accepted write request)
  // ---------------------------------------------------------------
  logic accept;
  assign accept = req_valid & req_ready;

  always_ff @(posedge clk) begin : bank_write
    if (accept && req.wr_en) begin : bank_write_exec
      mem[word_addr] <= req.data;
    end : bank_write_exec
  end : bank_write

  // ---------------------------------------------------------------
  // Pipeline stage 0: capture request
  // ---------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin : pipe_stage0
    if (!rst_n) begin : pipe_stage0_reset
      pipe_valid[0]   <= 1'b0;
      pipe_core_id[0] <= '0;
      pipe_req_id[0]  <= '0;
      pipe_is_read[0] <= 1'b0;
      pipe_rdata[0]   <= '0;
    end : pipe_stage0_reset
    else begin : pipe_stage0_active
      pipe_valid[0]   <= accept;
      pipe_core_id[0] <= req.core_id;
      pipe_req_id[0]  <= req.req_id;
      pipe_is_read[0] <= accept & ~req.wr_en;
      if (accept && !req.wr_en) begin : pipe_read_capture
        pipe_rdata[0] <= mem[word_addr];
      end : pipe_read_capture
      else begin : pipe_read_zero
        pipe_rdata[0] <= '0;
      end : pipe_read_zero
    end : pipe_stage0_active
  end : pipe_stage0

  // ---------------------------------------------------------------
  // Pipeline stages 1..PIPE_DEPTH-1
  // ---------------------------------------------------------------
  generate
    genvar gi;
    for (gi = 1; gi < PIPE_DEPTH; gi = gi + 1) begin : gen_pipe_stage
      always_ff @(posedge clk or negedge rst_n) begin : pipe_shift
        if (!rst_n) begin : pipe_shift_reset
          pipe_valid[gi]   <= 1'b0;
          pipe_core_id[gi] <= '0;
          pipe_req_id[gi]  <= '0;
          pipe_is_read[gi] <= 1'b0;
          pipe_rdata[gi]   <= '0;
        end : pipe_shift_reset
        else begin : pipe_shift_active
          pipe_valid[gi]   <= pipe_valid[gi-1];
          pipe_core_id[gi] <= pipe_core_id[gi-1];
          pipe_req_id[gi]  <= pipe_req_id[gi-1];
          pipe_is_read[gi] <= pipe_is_read[gi-1];
          pipe_rdata[gi]   <= pipe_rdata[gi-1];
        end : pipe_shift_active
      end : pipe_shift
    end : gen_pipe_stage
  endgenerate

  // ---------------------------------------------------------------
  // Response buffer: captures pipeline output
  // ---------------------------------------------------------------
  logic pipe_out_valid;
  assign pipe_out_valid = pipe_valid[PIPE_DEPTH-1];

  always_ff @(posedge clk or negedge rst_n) begin : resp_buf_update
    if (!rst_n) begin : resp_buf_reset
      resp_buf_valid  <= 1'b0;
      resp_buf        <= '0;
    end : resp_buf_reset
    else begin : resp_buf_active
      if (resp_buf_valid && resp_ready) begin : resp_buf_consumed
        // Response consumed downstream
        if (pipe_out_valid) begin : resp_buf_refill
          resp_buf_valid   <= 1'b1;
          resp_buf.data    <= pipe_rdata[PIPE_DEPTH-1];
          resp_buf.core_id <= pipe_core_id[PIPE_DEPTH-1];
          resp_buf.req_id  <= pipe_req_id[PIPE_DEPTH-1];
          resp_buf.error   <= 1'b0;
        end : resp_buf_refill
        else begin : resp_buf_clear
          resp_buf_valid <= 1'b0;
        end : resp_buf_clear
      end : resp_buf_consumed
      else if (!resp_buf_valid && pipe_out_valid) begin : resp_buf_load
        resp_buf_valid   <= 1'b1;
        resp_buf.data    <= pipe_rdata[PIPE_DEPTH-1];
        resp_buf.core_id <= pipe_core_id[PIPE_DEPTH-1];
        resp_buf.req_id  <= pipe_req_id[PIPE_DEPTH-1];
        resp_buf.error   <= 1'b0;
      end : resp_buf_load
    end : resp_buf_active
  end : resp_buf_update

  assign resp       = resp_buf;
  assign resp_valid = resp_buf_valid;

endmodule : tapestry_l2_bank
