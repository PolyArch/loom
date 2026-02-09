//===-- fabric_temporal_sw.sv - Temporal switch module ---------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Tag-matching crossbar switch. Routes tagged inputs to outputs based on
// tag-to-route table entries. Each table entry specifies:
//   {valid(1), tag(TAG_WIDTH), routes(K)} where K = popcount(CONNECTIVITY).
//
// Errors:
//   CFG_TEMPORAL_SW_DUP_TAG         - Duplicate valid tags in route table
//   CFG_TEMPORAL_SW_ROUTE_MULTI_OUT - Route entry drives same output multiple times
//   CFG_TEMPORAL_SW_ROUTE_MULTI_IN  - Multiple inputs routed to same output
//   RT_TEMPORAL_SW_NO_MATCH         - No table entry matches input tag
//   RT_TEMPORAL_SW_UNROUTED_INPUT   - Valid input has no route
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module fabric_temporal_sw #(
    parameter int NUM_INPUTS       = 2,
    parameter int NUM_OUTPUTS      = 2,
    parameter int DATA_WIDTH       = 32,
    parameter int TAG_WIDTH        = 4,
    parameter int NUM_ROUTE_TABLE  = 4,
    parameter logic [NUM_OUTPUTS*NUM_INPUTS-1:0] CONNECTIVITY = '1,
    localparam int PAYLOAD_WIDTH   = DATA_WIDTH + TAG_WIDTH,
    localparam int SAFE_PW         = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1
) (
    input  logic                                      clk,
    input  logic                                      rst_n,

    // Streaming inputs (packed, MSB = port[NUM_INPUTS-1])
    input  logic [NUM_INPUTS-1:0]                     in_valid,
    output logic [NUM_INPUTS-1:0]                     in_ready,
    input  logic [NUM_INPUTS*SAFE_PW-1:0]             in_data,

    // Streaming outputs
    output logic [NUM_OUTPUTS-1:0]                    out_valid,
    input  logic [NUM_OUTPUTS-1:0]                    out_ready,
    output logic [NUM_OUTPUTS*SAFE_PW-1:0]            out_data,

    // Configuration: route table entries
    input  logic [NUM_ROUTE_TABLE*(1+TAG_WIDTH+NUM_OUTPUTS)-1:0] cfg_data,

    // Error output
    output logic                                      error_valid,
    output logic [15:0]                               error_code
);

  // -----------------------------------------------------------------------
  // Elaboration-time parameter validation
  // -----------------------------------------------------------------------
  initial begin : param_check
    if (NUM_INPUTS < 1)
      $fatal(1, "COMP_TEMPORAL_SW_NUM_INPUTS: must be >= 1");
    if (NUM_OUTPUTS < 1)
      $fatal(1, "COMP_TEMPORAL_SW_NUM_OUTPUTS: must be >= 1");
    if (TAG_WIDTH < 1)
      $fatal(1, "COMP_TEMPORAL_SW_TAG_WIDTH: must be >= 1");
    if (NUM_ROUTE_TABLE < 1)
      $fatal(1, "COMP_TEMPORAL_SW_NUM_ROUTE_TABLE: must be >= 1");
  end

  // -----------------------------------------------------------------------
  // Unpack route table entries
  // Each entry: {valid(1), tag(TAG_WIDTH), routes(NUM_OUTPUTS)}
  // -----------------------------------------------------------------------
  localparam int ENTRY_WIDTH = 1 + TAG_WIDTH + NUM_OUTPUTS;

  logic [NUM_ROUTE_TABLE-1:0]                       rt_valid;
  logic [NUM_ROUTE_TABLE-1:0][TAG_WIDTH-1:0]        rt_tag;
  logic [NUM_ROUTE_TABLE-1:0][NUM_OUTPUTS-1:0]      rt_routes;

  always_comb begin : unpack_rt
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_ROUTE_TABLE; iter_var0 = iter_var0 + 1) begin : unpack
      rt_routes[iter_var0] = cfg_data[iter_var0 * ENTRY_WIDTH +: NUM_OUTPUTS];
      rt_tag[iter_var0]    = cfg_data[iter_var0 * ENTRY_WIDTH + NUM_OUTPUTS +: TAG_WIDTH];
      rt_valid[iter_var0]  = cfg_data[iter_var0 * ENTRY_WIDTH + NUM_OUTPUTS + TAG_WIDTH];
    end
  end

  // -----------------------------------------------------------------------
  // Per-input: extract tag, find matching route table entry
  // -----------------------------------------------------------------------
  logic [NUM_INPUTS-1:0][TAG_WIDTH-1:0]        in_tag;
  logic [NUM_INPUTS-1:0][SAFE_PW-1:0]          in_port_data;
  logic [NUM_INPUTS-1:0]                       in_match_found;
  logic [NUM_INPUTS-1:0][NUM_OUTPUTS-1:0]      in_routes;

  always_comb begin : tag_match
    integer iter_var0, iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : per_input
      in_port_data[iter_var0] = in_data[iter_var0 * SAFE_PW +: SAFE_PW];
      in_tag[iter_var0]       = in_port_data[iter_var0][DATA_WIDTH +: TAG_WIDTH];
      in_match_found[iter_var0] = 1'b0;
      in_routes[iter_var0]      = '0;
      for (iter_var1 = 0; iter_var1 < NUM_ROUTE_TABLE; iter_var1 = iter_var1 + 1) begin : search
        if (rt_valid[iter_var1] && (rt_tag[iter_var1] == in_tag[iter_var0])) begin : found
          in_match_found[iter_var0] = 1'b1;
          in_routes[iter_var0]      = rt_routes[iter_var1];
        end
      end
    end
  end

  // -----------------------------------------------------------------------
  // Output muxing: for each output, find which input is routed to it
  // Simple priority: lower-numbered input wins on contention
  // -----------------------------------------------------------------------
  always_comb begin : output_mux
    integer iter_var0, iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_OUTPUTS; iter_var0 = iter_var0 + 1) begin : per_output
      out_valid[iter_var0] = 1'b0;
      out_data[iter_var0 * SAFE_PW +: SAFE_PW] = '0;
      for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1 = iter_var1 + 1) begin : find_src
        if (in_valid[iter_var1] && in_match_found[iter_var1] &&
            in_routes[iter_var1][iter_var0] && !out_valid[iter_var0]) begin : route
          out_valid[iter_var0] = 1'b1;
          out_data[iter_var0 * SAFE_PW +: SAFE_PW] = in_port_data[iter_var1];
        end
      end
    end
  end

  // -----------------------------------------------------------------------
  // Ready logic: input is ready when all its target outputs are ready
  // -----------------------------------------------------------------------
  always_comb begin : ready_logic
    integer iter_var0, iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : per_input
      in_ready[iter_var0] = 1'b1;
      if (in_valid[iter_var0] && in_match_found[iter_var0]) begin : check_outputs
        for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1 = iter_var1 + 1) begin : per_out
          if (in_routes[iter_var0][iter_var1]) begin : routed
            if (!out_ready[iter_var1]) begin : not_ready
              in_ready[iter_var0] = 1'b0;
            end
          end
        end
      end else begin : no_route
        in_ready[iter_var0] = 1'b0;
      end
    end
  end

  // -----------------------------------------------------------------------
  // Error detection
  // -----------------------------------------------------------------------

  // CFG: duplicate valid tags
  logic cfg_dup_tag;
  always_comb begin : dup_tag_check
    integer iter_var0, iter_var1;
    cfg_dup_tag = 1'b0;
    for (iter_var0 = 0; iter_var0 < NUM_ROUTE_TABLE; iter_var0 = iter_var0 + 1) begin : outer
      for (iter_var1 = iter_var0 + 1; iter_var1 < NUM_ROUTE_TABLE; iter_var1 = iter_var1 + 1) begin : inner
        if (rt_valid[iter_var0] && rt_valid[iter_var1] &&
            (rt_tag[iter_var0] == rt_tag[iter_var1])) begin : dup
          cfg_dup_tag = 1'b1;
        end
      end
    end
  end

  // RT: no match for valid input
  logic rt_no_match;
  always_comb begin : no_match_check
    integer iter_var0;
    rt_no_match = 1'b0;
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : check
      if (in_valid[iter_var0] && !in_match_found[iter_var0]) begin : miss
        rt_no_match = 1'b1;
      end
    end
  end

  // Error latch
  always_ff @(posedge clk or negedge rst_n) begin : error_latch
    if (!rst_n) begin : reset
      error_valid <= 1'b0;
      error_code  <= 16'd0;
    end else if (!error_valid) begin : capture
      if (cfg_dup_tag) begin : dup_err
        error_valid <= 1'b1;
        error_code  <= CFG_TEMPORAL_SW_DUP_TAG;
      end else if (rt_no_match) begin : match_err
        error_valid <= 1'b1;
        error_code  <= RT_TEMPORAL_SW_NO_MATCH;
      end
    end
  end

endmodule
