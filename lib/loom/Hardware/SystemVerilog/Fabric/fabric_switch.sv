//===-- fabric_switch.sv - Parameterized switch module ---------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Configurable routing switch with CONNECTIVITY matrix and runtime route table.
// Data path is combinational; error reporting uses a latch (clk/rst_n).
//
// For semantics, see spec-fabric-switch.md.
// For error codes, see spec-fabric-error.md.
//
//===----------------------------------------------------------------------===//

module fabric_switch #(
    parameter int NUM_INPUTS  = 4,
    parameter int NUM_OUTPUTS = 4,
    parameter int DATA_WIDTH  = 32,
    parameter int TAG_WIDTH   = 0,
    parameter bit [NUM_OUTPUTS*NUM_INPUTS-1:0] CONNECTIVITY = '1,
    localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH,
    localparam int NUM_CONNECTED = $countones(CONNECTIVITY)
) (
    input  logic clk,
    input  logic rst_n,

    // Streaming input ports
    input  logic [NUM_INPUTS-1:0]                          in_valid,
    output logic [NUM_INPUTS-1:0]                          in_ready,
    input  logic [NUM_INPUTS-1:0][PAYLOAD_WIDTH-1:0]       in_data,

    // Streaming output ports
    output logic [NUM_OUTPUTS-1:0]                         out_valid,
    input  logic [NUM_OUTPUTS-1:0]                         out_ready,
    output logic [NUM_OUTPUTS-1:0][PAYLOAD_WIDTH-1:0]      out_data,

    // Configuration: one bit per connected position in CONNECTIVITY
    input  logic [NUM_CONNECTED-1:0]                       cfg_route_table,

    // Error reporting
    output logic        error_valid,
    output logic [15:0] error_code
);

  // -----------------------------------------------------------------------
  // Elaboration-time parameter validation (COMP_ errors)
  // -----------------------------------------------------------------------
  initial begin : param_check
    int iter_var0, iter_var1;
    if (NUM_INPUTS > 32 || NUM_OUTPUTS > 32)
      $fatal(1, "COMP_SWITCH_PORT_LIMIT: max 32 inputs/outputs");

    // Check each output row has at least one connection
    for (iter_var0 = 0; iter_var0 < NUM_OUTPUTS; iter_var0++) begin : check_row
      if (CONNECTIVITY[iter_var0*NUM_INPUTS +: NUM_INPUTS] == '0)
        $fatal(1, "COMP_SWITCH_ROW_EMPTY: output %0d has no connections", iter_var0);
    end

    // Check each input column has at least one connection
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0++) begin : check_col
      automatic bit has_conn = 0;
      for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1++) begin : scan_out
        has_conn |= CONNECTIVITY[iter_var1*NUM_INPUTS + iter_var0];
      end
      if (!has_conn)
        $fatal(1, "COMP_SWITCH_COL_EMPTY: input %0d has no connections", iter_var0);
    end
  end

  // -----------------------------------------------------------------------
  // Map compressed route_table bits to full [NUM_OUTPUTS][NUM_INPUTS] matrix
  // -----------------------------------------------------------------------
  logic [NUM_OUTPUTS-1:0][NUM_INPUTS-1:0] route_matrix;

  always_comb begin : expand_route
    int iter_var0, iter_var1;
    automatic int bit_idx = 0;
    for (iter_var0 = 0; iter_var0 < NUM_OUTPUTS; iter_var0++) begin : per_out
      for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1++) begin : per_in
        if (CONNECTIVITY[iter_var0*NUM_INPUTS + iter_var1]) begin : connected
          route_matrix[iter_var0][iter_var1] = cfg_route_table[bit_idx];
          bit_idx++;
        end else begin : unconnected
          route_matrix[iter_var0][iter_var1] = 1'b0;
        end
      end
    end
  end

  // -----------------------------------------------------------------------
  // Combinational data path: per-output mux + valid/ready forwarding
  // -----------------------------------------------------------------------

  // Per-input: which output is routing from this input (for ready generation)
  logic [NUM_INPUTS-1:0] input_routed;    // at least one output routes this input
  logic [NUM_INPUTS-1:0] input_dst_ready; // ready from the destination output

  always_comb begin : datapath
    int iter_var0, iter_var1;

    for (iter_var0 = 0; iter_var0 < NUM_OUTPUTS; iter_var0++) begin : out_mux
      out_valid[iter_var0] = 1'b0;
      out_data[iter_var0]  = '0;
      for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1++) begin : sel_in
        if (route_matrix[iter_var0][iter_var1]) begin : routed
          out_valid[iter_var0] = in_valid[iter_var1];
          out_data[iter_var0]  = in_data[iter_var1];
        end
      end
    end

    // Generate in_ready: input is ready when its routed output is ready
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0++) begin : in_rdy
      input_routed[iter_var0]    = 1'b0;
      input_dst_ready[iter_var0] = 1'b0;
      for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1++) begin : find_dst
        if (route_matrix[iter_var1][iter_var0]) begin : has_route
          input_routed[iter_var0]    = 1'b1;
          input_dst_ready[iter_var0] = out_ready[iter_var1];
        end
      end
    end

    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0++) begin : drive_rdy
      if (input_routed[iter_var0])
        in_ready[iter_var0] = input_dst_ready[iter_var0];
      else
        in_ready[iter_var0] = 1'b0;
    end
  end

  // -----------------------------------------------------------------------
  // Runtime error detection
  // -----------------------------------------------------------------------
  logic        err_detect;
  logic [15:0] err_code_comb;

  always_comb begin : err_check
    int iter_var0, iter_var1;
    err_detect    = 1'b0;
    err_code_comb = 16'hFFFF; // sentinel: no error detected yet

    // CFG_SWITCH_ROUTE_MULTI_OUT (code 1):
    // Per-output: check if more than one route bit is set
    for (iter_var0 = 0; iter_var0 < NUM_OUTPUTS; iter_var0++) begin : chk_multi_out
      if ($countones(route_matrix[iter_var0]) > 1) begin : err_multi_out
        err_detect = 1'b1;
        if (16'd1 < err_code_comb)
          err_code_comb = 16'd1;
      end
    end

    // CFG_SWITCH_ROUTE_MULTI_IN (code 2):
    // Per-input: check if routed to more than one output
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0++) begin : chk_multi_in
      automatic int route_count = 0;
      for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1++) begin : count_routes
        route_count += int'(route_matrix[iter_var1][iter_var0]);
      end
      if (route_count > 1) begin : err_multi_in
        err_detect = 1'b1;
        if (16'd2 < err_code_comb)
          err_code_comb = 16'd2;
      end
    end

    // RT_SWITCH_UNROUTED_INPUT (code 262):
    // Per-input: if valid and has physical connectivity but no enabled route
    for (iter_var0 = 0; iter_var0 < NUM_INPUTS; iter_var0++) begin : chk_unrouted
      automatic bit has_phys = 1'b0;
      for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1++) begin : scan_conn
        has_phys |= CONNECTIVITY[iter_var1*NUM_INPUTS + iter_var0];
      end
      if (has_phys && in_valid[iter_var0] && !input_routed[iter_var0]) begin : err_unrouted
        err_detect = 1'b1;
        if (16'd262 < err_code_comb)
          err_code_comb = 16'd262;
      end
    end

    // Replace sentinel with 0 when no error was detected
    if (!err_detect)
      err_code_comb = 16'd0;
  end

  // Error latch: once set, stays set (fatal, no recovery)
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
