//===-- fabric_temporal_sw.sv - Temporal switch module ---------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Tag-matching crossbar switch. Routes tagged inputs to outputs based on
// tag-to-route table entries. Each table entry specifies:
//   {routes(NUM_CONNECTED), tag(TAG_WIDTH), valid(1)}
// where NUM_CONNECTED = $countones(CONNECTIVITY).
//
// When multiple inputs contend for the same output, round-robin arbitration
// selects the winner (per-output priority pointer, starting from port 0).
//
// Errors:
//   CFG_TEMPORAL_SW_ROUTE_SAME_TAG_INPUTS_TO_SAME_OUTPUT - Output selects >1 input in a slot
//   CFG_TEMPORAL_SW_DUP_TAG         - Duplicate valid tags in route table
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
    parameter bit [NUM_OUTPUTS*NUM_INPUTS-1:0] CONNECTIVITY = '1,
    localparam int PAYLOAD_WIDTH   = DATA_WIDTH + TAG_WIDTH,
    localparam int SAFE_PW         = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1,
    localparam int NUM_CONNECTED   = $countones(CONNECTIVITY)
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

    // Configuration: route table entries (compressed)
    input  logic [NUM_ROUTE_TABLE*(1+TAG_WIDTH+NUM_CONNECTED)-1:0] cfg_data,

    // Error output
    output logic                                      error_valid,
    output logic [15:0]                               error_code
);

  // -----------------------------------------------------------------------
  // Elaboration-time parameter validation
  // -----------------------------------------------------------------------
  initial begin : param_check
    if (NUM_INPUTS < 1)
      $fatal(1, "CPL_TEMPORAL_SW_NUM_INPUTS: must be >= 1");
    if (NUM_OUTPUTS < 1)
      $fatal(1, "CPL_TEMPORAL_SW_NUM_OUTPUTS: must be >= 1");
    if (TAG_WIDTH < 1)
      $fatal(1, "CPL_TEMPORAL_SW_TAG_WIDTH: must be >= 1");
    if (NUM_ROUTE_TABLE < 1)
      $fatal(1, "CPL_TEMPORAL_SW_NUM_ROUTE_TABLE: must be >= 1");
  end

  // -----------------------------------------------------------------------
  // Unpack route table entries (compressed)
  // Each entry (MSB -> LSB): {routes(NUM_CONNECTED), tag(TAG_WIDTH), valid(1)}
  // -----------------------------------------------------------------------
  localparam int ENTRY_WIDTH = 1 + TAG_WIDTH + NUM_CONNECTED;

  logic [NUM_ROUTE_TABLE-1:0]                       rt_valid;
  logic [NUM_ROUTE_TABLE-1:0][TAG_WIDTH-1:0]        rt_tag;
  logic [NUM_ROUTE_TABLE-1:0][NUM_CONNECTED-1:0]    rt_routes_compressed;

  always_comb begin : unpack_rt
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_ROUTE_TABLE; iter_var0 = iter_var0 + 1) begin : unpack
      rt_valid[iter_var0]  = cfg_data[iter_var0 * ENTRY_WIDTH];
      rt_tag[iter_var0]    = cfg_data[iter_var0 * ENTRY_WIDTH + 1 +: TAG_WIDTH];
      rt_routes_compressed[iter_var0] =
          cfg_data[iter_var0 * ENTRY_WIDTH + 1 + TAG_WIDTH +: NUM_CONNECTED];
    end
  end

  // -----------------------------------------------------------------------
  // Expand compressed route table to full [NUM_ROUTE_TABLE][NUM_OUTPUTS][NUM_INPUTS]
  // Each compressed bit maps to one connected (output,input) edge in row-major order.
  // -----------------------------------------------------------------------
  logic [NUM_ROUTE_TABLE-1:0][NUM_OUTPUTS-1:0][NUM_INPUTS-1:0] rt_routes;

  always_comb begin : expand_routes
    integer iter_var0, iter_var1, iter_var2;
    for (iter_var0 = 0; iter_var0 < NUM_ROUTE_TABLE; iter_var0 = iter_var0 + 1) begin : per_entry
      automatic int bit_idx = 0;
      for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1 = iter_var1 + 1) begin : per_out
        for (iter_var2 = 0; iter_var2 < NUM_INPUTS; iter_var2 = iter_var2 + 1) begin : per_in
          if (CONNECTIVITY[iter_var1 * NUM_INPUTS + iter_var2]) begin : connected
            rt_routes[iter_var0][iter_var1][iter_var2] = rt_routes_compressed[iter_var0][bit_idx];
            bit_idx++;
          end else begin : unconnected
            rt_routes[iter_var0][iter_var1][iter_var2] = 1'b0;
          end
        end
      end
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
    integer iter_var0, iter_var1, iter_var2;
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : per_input
      in_port_data[iter_var0] = in_data[iter_var0 * SAFE_PW +: SAFE_PW];
      in_tag[iter_var0]       = in_port_data[iter_var0][DATA_WIDTH +: TAG_WIDTH];
      in_match_found[iter_var0] = 1'b0;
      in_routes[iter_var0]      = '0;
      for (iter_var1 = 0; iter_var1 < NUM_ROUTE_TABLE; iter_var1 = iter_var1 + 1) begin : search
        if (rt_valid[iter_var1] && (rt_tag[iter_var1] == in_tag[iter_var0])) begin : found
          in_match_found[iter_var0] = 1'b1;
          // Extract this input's routes from the full route matrix
          for (iter_var2 = 0; iter_var2 < NUM_OUTPUTS; iter_var2 = iter_var2 + 1) begin : extract
            in_routes[iter_var0][iter_var2] = rt_routes[iter_var1][iter_var2][iter_var0];
          end
        end
      end
    end
  end

  // -----------------------------------------------------------------------
  // Round-robin arbitration state: one priority pointer per output
  // -----------------------------------------------------------------------
  localparam int RR_PTR_W = (NUM_INPUTS > 1) ? $clog2(NUM_INPUTS) : 1;
  logic [NUM_OUTPUTS-1:0][RR_PTR_W-1:0] rr_ptr;

  // -----------------------------------------------------------------------
  // Output muxing: for each output, round-robin among contending inputs.
  // Supports broadcast: one input can route to multiple outputs per slot.
  // -----------------------------------------------------------------------
  logic [NUM_INPUTS-1:0]  input_routed;
  logic [NUM_INPUTS-1:0]  input_dst_ready;
  // Per-output: which input won arbitration (combinational)
  logic [NUM_OUTPUTS-1:0][RR_PTR_W-1:0] arb_winner;
  logic [NUM_OUTPUTS-1:0]               arb_valid;
  // Per-input: broadcast won (won arb for ALL broadcast targets, no ready check)
  logic [NUM_INPUTS-1:0]                broadcast_won;
  // Per-input: broadcast OK (won arb + all targets ready, for in_ready gating)
  logic [NUM_INPUTS-1:0]                broadcast_ok;

  always_comb begin : output_mux
    integer iter_var0, iter_var1, iter_var2;

    // Arbitration: per-output, determine winner among contending inputs
    for (iter_var0 = 0; iter_var0 < NUM_OUTPUTS; iter_var0 = iter_var0 + 1) begin : per_output
      out_data[iter_var0 * SAFE_PW +: SAFE_PW] = '0;
      arb_winner[iter_var0] = '0;
      arb_valid[iter_var0]  = 1'b0;
      // Round-robin scan: start from rr_ptr[output], wrap around
      for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1 = iter_var1 + 1) begin : rr_scan
        iter_var2 = (int'(rr_ptr[iter_var0]) + iter_var1) % NUM_INPUTS;
        if (!arb_valid[iter_var0] &&
            in_valid[iter_var2] && in_match_found[iter_var2] &&
            in_routes[iter_var2][iter_var0] &&
            CONNECTIVITY[iter_var0 * NUM_INPUTS + iter_var2]) begin : winner
          arb_valid[iter_var0]  = 1'b1;
          arb_winner[iter_var0] = iter_var2[RR_PTR_W-1:0];
          out_data[iter_var0 * SAFE_PW +: SAFE_PW] = in_port_data[iter_var2];
        end
      end
    end

    // Broadcast-won: for each input, verify it won arbitration for ALL its
    // broadcast targets (no ready check -- avoids combinational loop through
    // downstream consumers' ready paths).
    // Broadcast-ok: same but also requires all targets' out_ready (for in_ready).
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : bcast_chk
      input_routed[iter_var0] = 1'b0;
      broadcast_won[iter_var0] = 1'b1; // AND identity
      broadcast_ok[iter_var0]  = 1'b1; // AND identity
      for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1 = iter_var1 + 1) begin : per_tgt
        if (in_routes[iter_var0][iter_var1]) begin : has_route
          input_routed[iter_var0] = 1'b1;
          if (!(arb_valid[iter_var1] &&
                (int'(arb_winner[iter_var1]) == iter_var0))) begin : arb_fail
            broadcast_won[iter_var0] = 1'b0;
            broadcast_ok[iter_var0]  = 1'b0;
          end else if (!out_ready[iter_var1]) begin : not_ready
            broadcast_ok[iter_var0] = 1'b0;
          end
        end
      end
    end

    // Gate out_valid: winner won arbitration for ALL its broadcast targets.
    // No out_ready dependency in out_valid avoids combinational loops.
    // Atomic broadcast is still guaranteed because in_ready = broadcast_ok
    // (includes ready check), so the source only advances when ALL targets
    // have consumed.
    for (iter_var0 = 0; iter_var0 < NUM_OUTPUTS; iter_var0 = iter_var0 + 1) begin : gate_valid
      if (arb_valid[iter_var0])
        out_valid[iter_var0] = broadcast_won[arb_winner[iter_var0]];
      else
        out_valid[iter_var0] = 1'b0;
    end

    // Ready logic: input is ready when it has a match, is routed, and
    // broadcast is OK (won arb for all targets and all are ready).
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : drive_rdy
      if (in_valid[iter_var0] && in_match_found[iter_var0] &&
          input_routed[iter_var0])
        in_ready[iter_var0] = broadcast_ok[iter_var0];
      else
        in_ready[iter_var0] = 1'b0;
    end
  end

  // Round-robin pointer update: advance past winner on successful handshake
  always_ff @(posedge clk or negedge rst_n) begin : rr_update
    integer iter_var0;
    if (!rst_n) begin : reset
      for (iter_var0 = 0; iter_var0 < NUM_OUTPUTS; iter_var0 = iter_var0 + 1) begin : init_ptr
        rr_ptr[iter_var0] <= '0;
      end
    end else begin : advance
      for (iter_var0 = 0; iter_var0 < NUM_OUTPUTS; iter_var0 = iter_var0 + 1) begin : per_output
        if (arb_valid[iter_var0] && out_valid[iter_var0] && out_ready[iter_var0]) begin : handshake
          rr_ptr[iter_var0] <= (arb_winner[iter_var0] + 1) % NUM_INPUTS;
        end
      end
    end
  end

  // -----------------------------------------------------------------------
  // Error detection
  // -----------------------------------------------------------------------
  logic        err_detect;
  logic [15:0] err_code_comb;

  always_comb begin : err_check
    integer iter_var0, iter_var1;
    err_detect    = 1'b0;
    err_code_comb = 16'hFFFF;

    // CFG_TEMPORAL_SW_ROUTE_SAME_TAG_INPUTS_TO_SAME_OUTPUT (code 4):
    // Per slot, each output selects at most 1 input.
    // (Broadcast is allowed: one input may route to multiple outputs per slot.)
    for (iter_var0 = 0; iter_var0 < NUM_ROUTE_TABLE; iter_var0 = iter_var0 + 1) begin : chk_same_tag_inputs
      if (rt_valid[iter_var0]) begin : valid_slot
        for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1 = iter_var1 + 1) begin : per_out
          automatic int in_count = 0;
          for (integer iter_var2 = 0; iter_var2 < NUM_INPUTS; iter_var2 = iter_var2 + 1) begin : cnt_in
            in_count += int'(rt_routes[iter_var0][iter_var1][iter_var2]);
          end
          if (in_count > 1) begin : err_same_tag_inputs
            err_detect = 1'b1;
            if (CFG_TEMPORAL_SW_ROUTE_SAME_TAG_INPUTS_TO_SAME_OUTPUT < err_code_comb)
              err_code_comb = CFG_TEMPORAL_SW_ROUTE_SAME_TAG_INPUTS_TO_SAME_OUTPUT;
          end
        end
      end
    end

    // CFG_TEMPORAL_SW_DUP_TAG: duplicate valid tags
    for (iter_var0 = 0; iter_var0 < NUM_ROUTE_TABLE; iter_var0 = iter_var0 + 1) begin : chk_dup_outer
      for (iter_var1 = iter_var0 + 1; iter_var1 < NUM_ROUTE_TABLE; iter_var1 = iter_var1 + 1) begin : chk_dup_inner
        if (rt_valid[iter_var0] && rt_valid[iter_var1] &&
            (rt_tag[iter_var0] == rt_tag[iter_var1])) begin : dup
          err_detect = 1'b1;
          if (CFG_TEMPORAL_SW_DUP_TAG < err_code_comb)
            err_code_comb = CFG_TEMPORAL_SW_DUP_TAG;
        end
      end
    end

    // RT_TEMPORAL_SW_NO_MATCH: valid input with no matching tag
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : chk_no_match
      if (in_valid[iter_var0] && !in_match_found[iter_var0]) begin : miss
        err_detect = 1'b1;
        if (RT_TEMPORAL_SW_NO_MATCH < err_code_comb)
          err_code_comb = RT_TEMPORAL_SW_NO_MATCH;
      end
    end

    // RT_TEMPORAL_SW_UNROUTED_INPUT: valid input has match but no enabled route
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0 = iter_var0 + 1) begin : chk_unrouted
      if (in_valid[iter_var0] && in_match_found[iter_var0] &&
          !input_routed[iter_var0]) begin : unrouted
        err_detect = 1'b1;
        if (RT_TEMPORAL_SW_UNROUTED_INPUT < err_code_comb)
          err_code_comb = RT_TEMPORAL_SW_UNROUTED_INPUT;
      end
    end

    if (!err_detect)
      err_code_comb = 16'd0;
  end

  // Error latch: once set, stays set
  always_ff @(posedge clk or negedge rst_n) begin : error_latch
    if (!rst_n) begin : reset
      error_valid <= 1'b0;
      error_code  <= 16'd0;
    end else if (!error_valid && err_detect) begin : capture
      error_valid <= 1'b1;
      error_code  <= err_code_comb;
    end
  end

endmodule
