//===-- fabric_pe.sv - Compute PE skeleton --------------------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Parameterized Processing Element with a body region filled by exportSV.
// The region between BEGIN/END PE BODY markers is replaced with operation
// module instantiations.
//
// Micro-architecture (arith/math/llvm bodies):
//   [inputs] -> [combinational body] -> [shift-register pipeline] -> [outputs]
//
// Ready propagation is combinational (zero delay): downstream out_ready
// passes through all shift-register stages to in_ready.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module fabric_pe #(
    parameter int NUM_INPUTS   = 2,
    parameter int NUM_OUTPUTS  = 1,
    parameter int DATA_WIDTH   = 32,
    parameter int TAG_WIDTH    = 0,
    parameter int LATENCY_TYP  = 1,
    localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH,
    localparam int SAFE_PW       = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1,
    localparam int SAFE_DW       = (DATA_WIDTH > 0) ? DATA_WIDTH : 1,
    // Config: output tags (one per output, TAG_WIDTH bits each)
    localparam int TAG_CFG_BITS  = (TAG_WIDTH > 0) ? NUM_OUTPUTS * TAG_WIDTH : 0,
    localparam int CONFIG_WIDTH  = (TAG_CFG_BITS > 0) ? TAG_CFG_BITS : 0
) (
    input  logic                                  clk,
    input  logic                                  rst_n,

    // Streaming inputs (packed arrays, MSB = port[NUM_INPUTS-1])
    input  logic [NUM_INPUTS-1:0]                 in_valid,
    output logic [NUM_INPUTS-1:0]                 in_ready,
    input  logic [NUM_INPUTS-1:0][SAFE_PW-1:0]    in_data,

    // Streaming outputs
    output logic [NUM_OUTPUTS-1:0]                out_valid,
    input  logic [NUM_OUTPUTS-1:0]                out_ready,
    output logic [NUM_OUTPUTS-1:0][SAFE_PW-1:0]   out_data,

    // Configuration
    input  logic [CONFIG_WIDTH > 0 ? CONFIG_WIDTH-1 : 0 : 0] cfg_data
);

  // -----------------------------------------------------------------------
  // Elaboration-time parameter validation
  // -----------------------------------------------------------------------
  initial begin : param_check
    if (NUM_INPUTS < 1)
      $fatal(1, "COMP_PE_NUM_INPUTS: NUM_INPUTS must be >= 1");
    if (NUM_OUTPUTS < 1)
      $fatal(1, "COMP_PE_NUM_OUTPUTS: NUM_OUTPUTS must be >= 1");
    if (DATA_WIDTH < 1)
      $fatal(1, "COMP_PE_DATA_WIDTH: DATA_WIDTH must be >= 1");
    if (LATENCY_TYP < 0)
      $fatal(1, "COMP_PE_LATENCY: LATENCY_TYP must be >= 0");
  end

  // -----------------------------------------------------------------------
  // Tag stripping: extract value portion from each input
  // -----------------------------------------------------------------------
  logic [NUM_INPUTS-1:0][SAFE_DW-1:0] in_value;

  generate
    if (TAG_WIDTH > 0) begin : g_tag_strip
      genvar gi;
      for (gi = 0; gi < NUM_INPUTS; gi++) begin : g_strip
        assign in_value[gi] = in_data[gi][DATA_WIDTH-1:0];
      end
    end else begin : g_no_tag
      genvar gi;
      for (gi = 0; gi < NUM_INPUTS; gi++) begin : g_pass
        assign in_value[gi] = in_data[gi][DATA_WIDTH-1:0];
      end
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Input handshake: all inputs must be valid to fire
  // -----------------------------------------------------------------------
  logic all_in_valid;
  assign all_in_valid = &in_valid;

  // Ready gated by downstream ready (combinational, no buffering)
  logic pipeline_ready;

  generate
    genvar gi_rdy;
    for (gi_rdy = 0; gi_rdy < NUM_INPUTS; gi_rdy++) begin : g_in_ready
      assign in_ready[gi_rdy] = pipeline_ready && all_in_valid;
    end
  endgenerate

  // -----------------------------------------------------------------------
  // Body region: combinational operation (filled by exportSV)
  // -----------------------------------------------------------------------
  logic [NUM_OUTPUTS-1:0][SAFE_DW-1:0] body_result;

  // ===== BEGIN PE BODY =====
  // (replaced by exportSV based on MLIR body)
  // ===== END PE BODY =====

  // -----------------------------------------------------------------------
  // Latency shift-register pipeline
  // -----------------------------------------------------------------------
  generate
    if (LATENCY_TYP > 0) begin : g_pipeline
      // Shift register: LATENCY_TYP stages of {valid, data[NUM_OUTPUTS]}
      logic [LATENCY_TYP-1:0]                               sr_valid;
      logic [LATENCY_TYP-1:0][NUM_OUTPUTS-1:0][SAFE_DW-1:0] sr_data;

      // All output consumers must be ready for pipeline to advance
      logic all_out_ready;
      assign all_out_ready = &out_ready;
      assign pipeline_ready = all_out_ready;

      always_ff @(posedge clk or negedge rst_n) begin : shift_reg
        if (!rst_n) begin : reset
          integer iter_var0;
          for (iter_var0 = 0; iter_var0 < LATENCY_TYP; iter_var0 = iter_var0 + 1) begin : clr
            sr_valid[iter_var0] <= 1'b0;
          end
        end else if (all_out_ready) begin : advance
          integer iter_var0;
          // Stage 0 gets body output
          sr_valid[0] <= all_in_valid;
          sr_data[0]  <= body_result;
          // Subsequent stages shift
          for (iter_var0 = 1; iter_var0 < LATENCY_TYP; iter_var0 = iter_var0 + 1) begin : shift
            sr_valid[iter_var0] <= sr_valid[iter_var0 - 1];
            sr_data[iter_var0]  <= sr_data[iter_var0 - 1];
          end
        end
      end

      // Output from last stage
      logic                               last_valid;
      logic [NUM_OUTPUTS-1:0][SAFE_DW-1:0] last_data;
      assign last_valid = sr_valid[LATENCY_TYP - 1];
      assign last_data  = sr_data[LATENCY_TYP - 1];

      // Tag attachment and output assembly
      genvar go;
      for (go = 0; go < NUM_OUTPUTS; go++) begin : g_out
        if (TAG_WIDTH > 0) begin : g_tagged
          logic [TAG_WIDTH-1:0] output_tag;
          assign output_tag = cfg_data[go * TAG_WIDTH +: TAG_WIDTH];
          assign out_data[go]  = {output_tag, last_data[go]};
        end else begin : g_native
          assign out_data[go]  = last_data[go];
        end
        assign out_valid[go] = last_valid;
      end

    end else begin : g_bypass
      // LATENCY_TYP = 0: pure combinational passthrough
      logic all_out_ready;
      assign all_out_ready = &out_ready;
      assign pipeline_ready = all_out_ready;

      genvar go;
      for (go = 0; go < NUM_OUTPUTS; go++) begin : g_out
        if (TAG_WIDTH > 0) begin : g_tagged
          logic [TAG_WIDTH-1:0] output_tag;
          assign output_tag = cfg_data[go * TAG_WIDTH +: TAG_WIDTH];
          assign out_data[go]  = {output_tag, body_result[go]};
        end else begin : g_native
          assign out_data[go]  = body_result[go];
        end
        assign out_valid[go] = all_in_valid;
      end
    end
  endgenerate

endmodule
