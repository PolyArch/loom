//===-- fabric_memory_bridge.sv - Memory port bridge ------------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Bridges the ADG-level expanded per-tag logical ports to the singular
// tagged interface used by fabric_memory / fabric_extmemory.
//
// Input side:  Each expanded logical port is double-buffered, then a
//              priority arbiter selects one buffered request per cycle
//              to forward on the singular memory port.
//
// Output side: The singular memory output is routed by tag to one of
//              the expanded output ports, each backed by a double buffer.
//
// When LD_COUNT <= 1 and ST_COUNT <= 1 (TAG_WIDTH = 0) the bridge
// degenerates into simple double-buffer passthrough with no mux/demux.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module fabric_memory_bridge #(
    parameter int LD_COUNT   = 1,
    parameter int ST_COUNT   = 1,
    parameter int ADDR_WIDTH = `FABRIC_ADDR_BIT_WIDTH,
    parameter int ELEM_WIDTH = 32,
    parameter int TAG_WIDTH  = 0,

    localparam int ADDR_PW = (ADDR_WIDTH + TAG_WIDTH > 0) ? ADDR_WIDTH + TAG_WIDTH : 1,
    localparam int ELEM_PW = (ELEM_WIDTH + TAG_WIDTH > 0) ? ELEM_WIDTH + TAG_WIDTH : 1,
    localparam int DONE_PW = (TAG_WIDTH > 0) ? TAG_WIDTH : 1,
    localparam int SAFE_TW = (TAG_WIDTH > 0) ? TAG_WIDTH : 1,
    localparam int SAFE_LD = (LD_COUNT > 0) ? LD_COUNT : 1,
    localparam int SAFE_ST = (ST_COUNT > 0) ? ST_COUNT : 1
) (
    input  logic clk,
    input  logic rst_n,

    // --- Expanded input ports (from ADG logical ports) --------------------

    // Load address inputs [LD_COUNT ports]
    input  logic [SAFE_LD-1:0]               ld_addr_in_valid,
    output logic [SAFE_LD-1:0]               ld_addr_in_ready,
    input  logic [SAFE_LD-1:0][ADDR_PW-1:0]  ld_addr_in_data,

    // Store data inputs [ST_COUNT ports]
    input  logic [SAFE_ST-1:0]               st_data_in_valid,
    output logic [SAFE_ST-1:0]               st_data_in_ready,
    input  logic [SAFE_ST-1:0][ELEM_PW-1:0]  st_data_in_data,

    // Store address inputs [ST_COUNT ports]
    input  logic [SAFE_ST-1:0]               st_addr_in_valid,
    output logic [SAFE_ST-1:0]               st_addr_in_ready,
    input  logic [SAFE_ST-1:0][ADDR_PW-1:0]  st_addr_in_data,

    // --- Singular memory interface (to/from fabric_memory) ----------------

    output logic               ld_addr_valid,
    input  logic               ld_addr_ready,
    output logic [ADDR_PW-1:0] ld_addr_data,

    output logic               st_addr_valid,
    input  logic               st_addr_ready,
    output logic [ADDR_PW-1:0] st_addr_data,

    output logic               st_data_valid,
    input  logic               st_data_ready,
    output logic [ELEM_PW-1:0] st_data_data,

    input  logic               ld_data_valid,
    output logic               ld_data_ready,
    input  logic [ELEM_PW-1:0] ld_data_data,

    input  logic               ld_done_valid,
    output logic               ld_done_ready,
    input  logic [DONE_PW-1:0] ld_done_data,

    input  logic               st_done_valid,
    output logic               st_done_ready,
    input  logic [DONE_PW-1:0] st_done_data,

    // --- Expanded output ports (to ADG logical ports) ---------------------

    output logic [SAFE_LD-1:0]               ld_data_out_valid,
    input  logic [SAFE_LD-1:0]               ld_data_out_ready,
    output logic [SAFE_LD-1:0][ELEM_PW-1:0]  ld_data_out_data,

    output logic [SAFE_LD-1:0]               ld_done_out_valid,
    input  logic [SAFE_LD-1:0]               ld_done_out_ready,
    output logic [SAFE_LD-1:0][DONE_PW-1:0]  ld_done_out_data,

    output logic [SAFE_ST-1:0]               st_done_out_valid,
    input  logic [SAFE_ST-1:0]               st_done_out_ready,
    output logic [SAFE_ST-1:0][DONE_PW-1:0]  st_done_out_data
);

  // =====================================================================
  // Input side: double-buffer each expanded port, priority-mux to singular
  // =====================================================================

  // --- Load address path -----------------------------------------------
  generate
    if (LD_COUNT > 0) begin : gen_ld_addr
      logic [LD_COUNT-1:0]               buf_valid;
      logic [LD_COUNT-1:0]               buf_ready;
      logic [LD_COUNT-1:0][ADDR_PW-1:0]  buf_data;

      for (genvar g = 0; g < LD_COUNT; g++) begin : gen_buf
        fabric_double_buffer #(.WIDTH(ADDR_PW)) inst (
          .clk, .rst_n,
          .in_valid (ld_addr_in_valid[g]),
          .in_ready (ld_addr_in_ready[g]),
          .in_data  (ld_addr_in_data[g]),
          .out_valid(buf_valid[g]),
          .out_ready(buf_ready[g]),
          .out_data (buf_data[g])
        );
      end

      always_comb begin : ld_addr_select
        integer iter_var0;
        ld_addr_valid = 1'b0;
        ld_addr_data  = '0;
        for (iter_var0 = 0; iter_var0 < LD_COUNT; iter_var0++) begin : clear_rdy
          buf_ready[iter_var0] = 1'b0;
        end
        for (iter_var0 = 0; iter_var0 < LD_COUNT; iter_var0++) begin : pick
          if (buf_valid[iter_var0] && !ld_addr_valid) begin : selected
            ld_addr_valid = 1'b1;
            ld_addr_data  = buf_data[iter_var0];
            buf_ready[iter_var0] = ld_addr_ready;
          end
        end
      end
    end else begin : gen_no_ld_addr
      assign ld_addr_valid     = 1'b0;
      assign ld_addr_data      = '0;
      assign ld_addr_in_ready  = '0;
    end
  endgenerate

  // --- Store address path ----------------------------------------------
  generate
    if (ST_COUNT > 0) begin : gen_st_addr
      logic [ST_COUNT-1:0]               buf_valid;
      logic [ST_COUNT-1:0]               buf_ready;
      logic [ST_COUNT-1:0][ADDR_PW-1:0]  buf_data;

      for (genvar g = 0; g < ST_COUNT; g++) begin : gen_buf
        fabric_double_buffer #(.WIDTH(ADDR_PW)) inst (
          .clk, .rst_n,
          .in_valid (st_addr_in_valid[g]),
          .in_ready (st_addr_in_ready[g]),
          .in_data  (st_addr_in_data[g]),
          .out_valid(buf_valid[g]),
          .out_ready(buf_ready[g]),
          .out_data (buf_data[g])
        );
      end

      always_comb begin : st_addr_select
        integer iter_var0;
        st_addr_valid = 1'b0;
        st_addr_data  = '0;
        for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0++) begin : clear_rdy
          buf_ready[iter_var0] = 1'b0;
        end
        for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0++) begin : pick
          if (buf_valid[iter_var0] && !st_addr_valid) begin : selected
            st_addr_valid = 1'b1;
            st_addr_data  = buf_data[iter_var0];
            buf_ready[iter_var0] = st_addr_ready;
          end
        end
      end
    end else begin : gen_no_st_addr
      assign st_addr_valid     = 1'b0;
      assign st_addr_data      = '0;
      assign st_addr_in_ready  = '0;
    end
  endgenerate

  // --- Store data path -------------------------------------------------
  generate
    if (ST_COUNT > 0) begin : gen_st_data
      logic [ST_COUNT-1:0]               buf_valid;
      logic [ST_COUNT-1:0]               buf_ready;
      logic [ST_COUNT-1:0][ELEM_PW-1:0]  buf_data;

      for (genvar g = 0; g < ST_COUNT; g++) begin : gen_buf
        fabric_double_buffer #(.WIDTH(ELEM_PW)) inst (
          .clk, .rst_n,
          .in_valid (st_data_in_valid[g]),
          .in_ready (st_data_in_ready[g]),
          .in_data  (st_data_in_data[g]),
          .out_valid(buf_valid[g]),
          .out_ready(buf_ready[g]),
          .out_data (buf_data[g])
        );
      end

      always_comb begin : st_data_select
        integer iter_var0;
        st_data_valid = 1'b0;
        st_data_data  = '0;
        for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0++) begin : clear_rdy
          buf_ready[iter_var0] = 1'b0;
        end
        for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0++) begin : pick
          if (buf_valid[iter_var0] && !st_data_valid) begin : selected
            st_data_valid = 1'b1;
            st_data_data  = buf_data[iter_var0];
            buf_ready[iter_var0] = st_data_ready;
          end
        end
      end
    end else begin : gen_no_st_data
      assign st_data_valid     = 1'b0;
      assign st_data_data      = '0;
      assign st_data_in_ready  = '0;
    end
  endgenerate

  // =====================================================================
  // Output side: route singular output by tag to per-port double buffers
  // =====================================================================

  // --- Load data path --------------------------------------------------
  generate
    if (LD_COUNT > 0) begin : gen_ld_data_out
      logic [LD_COUNT-1:0]               buf_in_valid;
      logic [LD_COUNT-1:0]               buf_in_ready;
      logic [LD_COUNT-1:0][ELEM_PW-1:0]  buf_in_data;

      for (genvar g = 0; g < LD_COUNT; g++) begin : gen_buf
        fabric_double_buffer #(.WIDTH(ELEM_PW)) inst (
          .clk, .rst_n,
          .in_valid (buf_in_valid[g]),
          .in_ready (buf_in_ready[g]),
          .in_data  (buf_in_data[g]),
          .out_valid(ld_data_out_valid[g]),
          .out_ready(ld_data_out_ready[g]),
          .out_data (ld_data_out_data[g])
        );
      end

      if (TAG_WIDTH > 0) begin : gen_tagged
        always_comb begin : ld_data_route
          integer iter_var0;
          logic [SAFE_TW-1:0] tag;
          ld_data_ready = 1'b0;
          for (iter_var0 = 0; iter_var0 < LD_COUNT; iter_var0++) begin : clear_enq
            buf_in_valid[iter_var0] = 1'b0;
            buf_in_data[iter_var0]  = ld_data_data;
          end
          tag = ld_data_data[ELEM_WIDTH +: TAG_WIDTH];
          for (iter_var0 = 0; iter_var0 < LD_COUNT; iter_var0++) begin : route
            if (iter_var0[SAFE_TW-1:0] == tag) begin : match
              ld_data_ready = buf_in_ready[iter_var0];
              buf_in_valid[iter_var0] = ld_data_valid;
            end
          end
        end
      end else begin : gen_untagged
        always_comb begin : ld_data_route
          ld_data_ready    = buf_in_ready[0];
          buf_in_valid[0]  = ld_data_valid;
          buf_in_data[0]   = ld_data_data;
        end
      end
    end else begin : gen_no_ld_data
      assign ld_data_ready      = 1'b0;
      assign ld_data_out_valid  = '0;
      assign ld_data_out_data   = '0;
    end
  endgenerate

  // --- Load done path --------------------------------------------------
  generate
    if (LD_COUNT > 0) begin : gen_ld_done_out
      logic [LD_COUNT-1:0]               buf_in_valid;
      logic [LD_COUNT-1:0]               buf_in_ready;
      logic [LD_COUNT-1:0][DONE_PW-1:0]  buf_in_data;

      for (genvar g = 0; g < LD_COUNT; g++) begin : gen_buf
        fabric_double_buffer #(.WIDTH(DONE_PW)) inst (
          .clk, .rst_n,
          .in_valid (buf_in_valid[g]),
          .in_ready (buf_in_ready[g]),
          .in_data  (buf_in_data[g]),
          .out_valid(ld_done_out_valid[g]),
          .out_ready(ld_done_out_ready[g]),
          .out_data (ld_done_out_data[g])
        );
      end

      if (TAG_WIDTH > 0) begin : gen_tagged
        always_comb begin : ld_done_route
          integer iter_var0;
          logic [SAFE_TW-1:0] tag;
          ld_done_ready = 1'b0;
          for (iter_var0 = 0; iter_var0 < LD_COUNT; iter_var0++) begin : clear_enq
            buf_in_valid[iter_var0] = 1'b0;
            buf_in_data[iter_var0]  = ld_done_data;
          end
          tag = ld_done_data[0 +: TAG_WIDTH];
          for (iter_var0 = 0; iter_var0 < LD_COUNT; iter_var0++) begin : route
            if (iter_var0[SAFE_TW-1:0] == tag) begin : match
              ld_done_ready = buf_in_ready[iter_var0];
              buf_in_valid[iter_var0] = ld_done_valid;
            end
          end
        end
      end else begin : gen_untagged
        always_comb begin : ld_done_route
          ld_done_ready    = buf_in_ready[0];
          buf_in_valid[0]  = ld_done_valid;
          buf_in_data[0]   = ld_done_data;
        end
      end
    end else begin : gen_no_ld_done
      assign ld_done_ready      = 1'b0;
      assign ld_done_out_valid  = '0;
      assign ld_done_out_data   = '0;
    end
  endgenerate

  // --- Store done path -------------------------------------------------
  generate
    if (ST_COUNT > 0) begin : gen_st_done_out
      logic [ST_COUNT-1:0]               buf_in_valid;
      logic [ST_COUNT-1:0]               buf_in_ready;
      logic [ST_COUNT-1:0][DONE_PW-1:0]  buf_in_data;

      for (genvar g = 0; g < ST_COUNT; g++) begin : gen_buf
        fabric_double_buffer #(.WIDTH(DONE_PW)) inst (
          .clk, .rst_n,
          .in_valid (buf_in_valid[g]),
          .in_ready (buf_in_ready[g]),
          .in_data  (buf_in_data[g]),
          .out_valid(st_done_out_valid[g]),
          .out_ready(st_done_out_ready[g]),
          .out_data (st_done_out_data[g])
        );
      end

      if (TAG_WIDTH > 0) begin : gen_tagged
        always_comb begin : st_done_route
          integer iter_var0;
          logic [SAFE_TW-1:0] tag;
          st_done_ready = 1'b0;
          for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0++) begin : clear_enq
            buf_in_valid[iter_var0] = 1'b0;
            buf_in_data[iter_var0]  = st_done_data;
          end
          tag = st_done_data[0 +: TAG_WIDTH];
          for (iter_var0 = 0; iter_var0 < ST_COUNT; iter_var0++) begin : route
            if (iter_var0[SAFE_TW-1:0] == tag) begin : match
              st_done_ready = buf_in_ready[iter_var0];
              buf_in_valid[iter_var0] = st_done_valid;
            end
          end
        end
      end else begin : gen_untagged
        always_comb begin : st_done_route
          st_done_ready    = buf_in_ready[0];
          buf_in_valid[0]  = st_done_valid;
          buf_in_data[0]   = st_done_data;
        end
      end
    end else begin : gen_no_st_done
      assign st_done_ready      = 1'b0;
      assign st_done_out_valid  = '0;
      assign st_done_out_data   = '0;
    end
  endgenerate

endmodule
