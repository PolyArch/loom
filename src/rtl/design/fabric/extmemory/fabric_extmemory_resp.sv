// fabric_extmemory_resp.sv -- AXI response handler for external memory.
//
// Converts AXI R channel responses into load_data/load_done outputs,
// and AXI B channel responses into store_done outputs.
//
// Tag preservation: the AXI ID field carries the lane tag from the
// original request.  The response handler extracts the tag and
// attaches it to the corresponding output port.
//
// Output holding registers absorb backpressure from downstream.
// One response per channel per cycle is accepted.

module fabric_extmemory_resp #(
  parameter int unsigned DATA_WIDTH   = 32,
  parameter int unsigned TAG_WIDTH    = 4,
  parameter int unsigned AXI_ID_WIDTH = 4,
  // Derived parameters (used in port declarations)
  parameter int unsigned TAG_W        = (TAG_WIDTH > 0) ? TAG_WIDTH : 1
)(
  input  logic                    clk,
  input  logic                    rst_n,

  // --- AXI Read Data Channel (R) ---
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [AXI_ID_WIDTH-1:0] m_axi_rid,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic [DATA_WIDTH-1:0]  m_axi_rdata,
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [1:0]              m_axi_rresp,
  input  logic                    m_axi_rlast,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic                    m_axi_rvalid,
  output logic                    m_axi_rready,

  // --- AXI Write Response Channel (B) ---
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [AXI_ID_WIDTH-1:0] m_axi_bid,
  input  logic [1:0]              m_axi_bresp,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic                    m_axi_bvalid,
  output logic                    m_axi_bready,

  // --- Load data output ---
  output logic                    ld_data_valid,
  input  logic                    ld_data_ready,
  output logic [DATA_WIDTH-1:0]  ld_data_data,
  output logic [TAG_W-1:0]       ld_data_tag,

  // --- Load done output ---
  output logic                    ld_done_valid,
  input  logic                    ld_done_ready,
  output logic [TAG_W-1:0]       ld_done_tag,

  // --- Store done output ---
  output logic                    st_done_valid,
  input  logic                    st_done_ready,
  output logic [TAG_W-1:0]       st_done_tag,

  // --- Completion notification to top for outstanding tracking ---
  output logic                    ld_completed,
  output logic [TAG_W-1:0]       ld_completed_lane,
  output logic                    st_completed,
  output logic [TAG_W-1:0]       st_completed_lane
);

  // ---------------------------------------------------------------
  // Load response (AXI R -> load_data + load_done)
  // ---------------------------------------------------------------
  // Output holding registers.
  logic             ld_data_out_valid;
  logic [DATA_WIDTH-1:0] ld_data_out_data;
  logic [TAG_W-1:0] ld_data_out_tag;
  logic             ld_done_out_valid;
  logic [TAG_W-1:0] ld_done_out_tag;

  // Accept R channel when output register is free (or being drained).
  logic ld_out_can_accept;
  assign ld_out_can_accept = (!ld_data_out_valid || (ld_data_valid && ld_data_ready)) &&
                             (!ld_done_out_valid || (ld_done_valid && ld_done_ready));

  assign m_axi_rready = ld_out_can_accept;

  // Tag extracted from AXI ID.
  logic [TAG_W-1:0] r_tag;
  assign r_tag = m_axi_rid[TAG_W-1:0];

  always_ff @(posedge clk or negedge rst_n) begin : ld_resp_update
    if (!rst_n) begin : ld_resp_reset
      ld_data_out_valid <= 1'b0;
      ld_data_out_data  <= '0;
      ld_data_out_tag   <= '0;
      ld_done_out_valid <= 1'b0;
      ld_done_out_tag   <= '0;
    end : ld_resp_reset
    else begin : ld_resp_op

      // Accept new R channel response.
      if (m_axi_rvalid && m_axi_rready) begin : ld_resp_accept
        ld_data_out_valid <= 1'b1;
        ld_data_out_data  <= m_axi_rdata;
        ld_data_out_tag   <= r_tag;
        ld_done_out_valid <= 1'b1;
        ld_done_out_tag   <= r_tag;
      end : ld_resp_accept
      else begin : ld_resp_drain

        // Drain load data output.
        if (ld_data_out_valid && ld_data_ready) begin : ld_data_drain
          ld_data_out_valid <= 1'b0;
        end : ld_data_drain

        // Drain load done output.
        if (ld_done_out_valid && ld_done_ready) begin : ld_done_drain
          ld_done_out_valid <= 1'b0;
        end : ld_done_drain

      end : ld_resp_drain
    end : ld_resp_op
  end : ld_resp_update

  assign ld_data_valid = ld_data_out_valid;
  assign ld_data_data  = ld_data_out_data;
  assign ld_data_tag   = ld_data_out_tag;
  assign ld_done_valid = ld_done_out_valid;
  assign ld_done_tag   = ld_done_out_tag;

  // Completion notification for load.
  assign ld_completed      = m_axi_rvalid && m_axi_rready;
  assign ld_completed_lane = r_tag;

  // ---------------------------------------------------------------
  // Store response (AXI B -> store_done)
  // ---------------------------------------------------------------
  logic             st_done_out_valid;
  logic [TAG_W-1:0] st_done_out_tag;

  logic st_out_can_accept;
  assign st_out_can_accept = !st_done_out_valid || (st_done_valid && st_done_ready);

  assign m_axi_bready = st_out_can_accept;

  logic [TAG_W-1:0] b_tag;
  assign b_tag = m_axi_bid[TAG_W-1:0];

  always_ff @(posedge clk or negedge rst_n) begin : st_resp_update
    if (!rst_n) begin : st_resp_reset
      st_done_out_valid <= 1'b0;
      st_done_out_tag   <= '0;
    end : st_resp_reset
    else begin : st_resp_op

      if (m_axi_bvalid && m_axi_bready) begin : st_resp_accept
        st_done_out_valid <= 1'b1;
        st_done_out_tag   <= b_tag;
      end : st_resp_accept
      else begin : st_resp_drain
        if (st_done_out_valid && st_done_ready) begin : st_done_drain
          st_done_out_valid <= 1'b0;
        end : st_done_drain
      end : st_resp_drain

    end : st_resp_op
  end : st_resp_update

  assign st_done_valid = st_done_out_valid;
  assign st_done_tag   = st_done_out_tag;

  // Completion notification for store.
  assign st_completed      = m_axi_bvalid && m_axi_bready;
  assign st_completed_lane = b_tag;

endmodule : fabric_extmemory_resp
