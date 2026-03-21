// fabric_spatial_sw_decomp.sv -- Decomposable spatial switch.
//
// Splits the data bus into sub-lanes of DECOMPOSABLE_BITS width.
// Each sub-lane has its own independent mux and route configuration,
// allowing sub-word routing (e.g., routing different bytes of a wide
// bus from different input ports).
//
// Decomposable switches do not support tags (TAG_WIDTH == 0 enforced
// by the top-level wrapper).
//
// Config bitstream layout per sub-lane:
//   route_bits = popcount(CONNECTIVITY)  (same connectivity for all sub-lanes)
//   discard_bits = NUM_IN
//
// Total config bits = NUM_LANES * (route_bits + discard_bits)
//
// Sub-lanes are packed in ascending lane order (lane 0 first).
// Within each sub-lane, the layout matches fabric_spatial_sw_core.

module fabric_spatial_sw_decomp
  import fabric_pkg::*;
#(
  parameter int unsigned NUM_IN            = 2,
  parameter int unsigned NUM_OUT           = 2,
  parameter int unsigned DATA_WIDTH        = 32,
  parameter bit [NUM_OUT*NUM_IN-1:0] CONNECTIVITY = {NUM_OUT*NUM_IN{1'b1}},
  parameter int unsigned DECOMPOSABLE_BITS = 8
)(
  input  logic                    clk,
  input  logic                    rst_n,

  // --- Per-input handshake ---
  input  logic [NUM_IN-1:0]       in_valid,
  output logic [NUM_IN-1:0]       in_ready,
  input  logic [DATA_WIDTH-1:0]   in_data  [NUM_IN],

  // --- Per-output handshake ---
  output logic [NUM_OUT-1:0]      out_valid,
  input  logic [NUM_OUT-1:0]      out_ready,
  output logic [DATA_WIDTH-1:0]   out_data [NUM_OUT],

  // --- Config port (word-serial) ---
  input  logic                    cfg_valid,
  input  logic [31:0]             cfg_wdata,
  output logic                    cfg_ready
);

  // ---------------------------------------------------------------
  // Sub-lane geometry
  // ---------------------------------------------------------------
  localparam int unsigned NUM_LANES = DATA_WIDTH / DECOMPOSABLE_BITS;

  // Connectivity helpers (same functions as core).
  function automatic int unsigned count_connected();
    int unsigned cnt;
    integer iter_var0;
    cnt = 0;
    for (iter_var0 = 0; iter_var0 < NUM_OUT * NUM_IN; iter_var0 = iter_var0 + 1) begin : popcount_loop
      if (CONNECTIVITY[iter_var0])
        cnt = cnt + 1;
    end : popcount_loop
    return cnt;
  endfunction : count_connected

  localparam int unsigned ROUTE_BITS_PER_LANE = count_connected();
  localparam int unsigned DISCARD_BITS_PER_LANE = NUM_IN;
  localparam int unsigned CFG_BITS_PER_LANE = ROUTE_BITS_PER_LANE + DISCARD_BITS_PER_LANE;
  localparam int unsigned TOTAL_CFG_BITS = NUM_LANES * CFG_BITS_PER_LANE;
  localparam int unsigned CFG_WORDS = (TOTAL_CFG_BITS + 31) / 32;

  // ---------------------------------------------------------------
  // Config register storage
  // ---------------------------------------------------------------
  logic [TOTAL_CFG_BITS > 0 ? TOTAL_CFG_BITS-1 : 0 : 0] cfg_bits;
  logic [$clog2(CFG_WORDS > 1 ? CFG_WORDS : 2)-1:0] cfg_word_cnt;

  assign cfg_ready = 1'b1;

  always_ff @(posedge clk or negedge rst_n) begin : cfg_load
    if (!rst_n) begin : cfg_reset
      cfg_bits     <= '0;
      cfg_word_cnt <= '0;
    end : cfg_reset
    else begin : cfg_update
      if (cfg_valid && cfg_ready) begin : cfg_capture
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < 32; iter_var0 = iter_var0 + 1) begin : cfg_bit_loop
          if ((int'(cfg_word_cnt) * 32 + iter_var0) < TOTAL_CFG_BITS) begin : cfg_bit_valid
            cfg_bits[int'(cfg_word_cnt) * 32 + iter_var0] <= cfg_wdata[iter_var0];
          end : cfg_bit_valid
        end : cfg_bit_loop
        if (cfg_word_cnt == CFG_WORDS[$clog2(CFG_WORDS > 1 ? CFG_WORDS : 2)-1:0] - 1'b1) begin : cfg_wrap
          cfg_word_cnt <= '0;
        end : cfg_wrap
        else begin : cfg_incr
          cfg_word_cnt <= cfg_word_cnt + 1'b1;
        end : cfg_incr
      end : cfg_capture
    end : cfg_update
  end : cfg_load

  // ---------------------------------------------------------------
  // Per-lane sub-switch logic
  // ---------------------------------------------------------------
  // Each sub-lane has an independent route bitmap and discard vector.
  // The sub-lane mux selects DECOMPOSABLE_BITS-wide slices of the
  // input data bus.

  // Per-lane per-output: selected input index.
  logic [$clog2(NUM_IN > 1 ? NUM_IN : 2)-1:0] lane_src_idx [NUM_LANES][NUM_OUT];
  logic [NUM_OUT-1:0] lane_out_valid [NUM_LANES];
  // Per-lane per-input: ready signal.
  logic [NUM_IN-1:0] lane_in_ready [NUM_LANES];

  generate
    genvar gl;
    for (gl = 0; gl < NUM_LANES; gl = gl + 1) begin : gen_lane
      // Extract per-lane config bits.
      localparam int unsigned LANE_CFG_BASE = gl * CFG_BITS_PER_LANE;

      logic [ROUTE_BITS_PER_LANE > 0 ? ROUTE_BITS_PER_LANE-1 : 0 : 0] lane_route;
      logic [NUM_IN-1:0] lane_discard;

      if (ROUTE_BITS_PER_LANE > 0) begin : gen_lane_route_extract
        assign lane_route = cfg_bits[LANE_CFG_BASE + ROUTE_BITS_PER_LANE - 1 : LANE_CFG_BASE];
      end : gen_lane_route_extract
      else begin : gen_lane_route_zero
        assign lane_route = '0;
      end : gen_lane_route_zero

      assign lane_discard = cfg_bits[LANE_CFG_BASE + ROUTE_BITS_PER_LANE + NUM_IN - 1 :
                                     LANE_CFG_BASE + ROUTE_BITS_PER_LANE];

      // Build route_enabled for this lane (same structure as core).
      logic [NUM_IN-1:0] lane_route_enabled [NUM_OUT];

      always_comb begin : lane_build_route_enabled
        integer iter_var0;
        integer iter_var1;
        integer bit_idx;
        bit_idx = 0;
        for (iter_var0 = 0; iter_var0 < NUM_OUT; iter_var0 = iter_var0 + 1) begin : lre_out_loop
          lane_route_enabled[iter_var0] = '0;
          for (iter_var1 = 0; iter_var1 < NUM_IN; iter_var1 = iter_var1 + 1) begin : lre_in_loop
            if (CONNECTIVITY[iter_var0 * NUM_IN + iter_var1]) begin : lre_connected
              if (ROUTE_BITS_PER_LANE > 0) begin : lre_check_bit
                lane_route_enabled[iter_var0][iter_var1] = lane_route[bit_idx];
              end : lre_check_bit
              bit_idx = bit_idx + 1;
            end : lre_connected
          end : lre_in_loop
        end : lre_out_loop
      end : lane_build_route_enabled

      // Per-output priority select (decomp switches are non-tagged,
      // so no round-robin needed).
      always_comb begin : lane_output_select
        integer iter_var0;
        integer iter_var1;
        for (iter_var0 = 0; iter_var0 < NUM_OUT; iter_var0 = iter_var0 + 1) begin : los_out_loop
          lane_out_valid[gl][iter_var0] = 1'b0;
          lane_src_idx[gl][iter_var0] = '0;
          for (iter_var1 = 0; iter_var1 < NUM_IN; iter_var1 = iter_var1 + 1) begin : los_in_loop
            if (!lane_out_valid[gl][iter_var0] &&
                lane_route_enabled[iter_var0][iter_var1] &&
                in_valid[iter_var1] &&
                !lane_discard[iter_var1]) begin : los_hit
              lane_out_valid[gl][iter_var0] = 1'b1;
              lane_src_idx[gl][iter_var0] = iter_var1[$clog2(NUM_IN > 1 ? NUM_IN : 2)-1:0];
            end : los_hit
          end : los_in_loop
        end : los_out_loop
      end : lane_output_select

      // Per-input ready for this lane.
      // Input ready when: discarded OR all targets for this input
      // on this lane are ready.
      always_comb begin : lane_input_ready
        integer iter_var0;
        integer iter_var1;
        for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : lir_in_loop
          if (lane_discard[iter_var0]) begin : lir_discard
            lane_in_ready[gl][iter_var0] = 1'b1;
          end : lir_discard
          else begin : lir_check
            // Count targets on this lane for this input.
            automatic logic has_target;
            automatic logic all_ok;
            has_target = 1'b0;
            all_ok = 1'b1;
            for (iter_var1 = 0; iter_var1 < NUM_OUT; iter_var1 = iter_var1 + 1) begin : lir_out_loop
              if (lane_route_enabled[iter_var1][iter_var0]) begin : lir_target
                has_target = 1'b1;
                // Check: this output selected this input AND output is ready.
                if (!(lane_out_valid[gl][iter_var1] &&
                      lane_src_idx[gl][iter_var1] == iter_var0[$clog2(NUM_IN > 1 ? NUM_IN : 2)-1:0] &&
                      out_ready[iter_var1])) begin : lir_not_ready
                  all_ok = 1'b0;
                end : lir_not_ready
              end : lir_target
            end : lir_out_loop
            lane_in_ready[gl][iter_var0] = has_target & all_ok;
          end : lir_check
        end : lir_in_loop
      end : lane_input_ready

      // Wire output data sub-lane.
      genvar go;
      for (go = 0; go < NUM_OUT; go = go + 1) begin : gen_lane_data
        assign out_data[go][(gl+1)*DECOMPOSABLE_BITS-1 : gl*DECOMPOSABLE_BITS] =
          in_data[lane_src_idx[gl][go]][(gl+1)*DECOMPOSABLE_BITS-1 : gl*DECOMPOSABLE_BITS];
      end : gen_lane_data

    end : gen_lane
  endgenerate

  // ---------------------------------------------------------------
  // Aggregate output valid: output is valid only when ALL sub-lanes
  // have a valid source for that output.
  // ---------------------------------------------------------------
  always_comb begin : aggregate_out_valid
    integer iter_var0;
    integer iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_OUT; iter_var0 = iter_var0 + 1) begin : aov_out_loop
      out_valid[iter_var0] = 1'b1;
      for (iter_var1 = 0; iter_var1 < NUM_LANES; iter_var1 = iter_var1 + 1) begin : aov_lane_loop
        if (!lane_out_valid[iter_var1][iter_var0]) begin : aov_invalid
          out_valid[iter_var0] = 1'b0;
        end : aov_invalid
      end : aov_lane_loop
    end : aov_out_loop
  end : aggregate_out_valid

  // ---------------------------------------------------------------
  // Aggregate input ready: input is ready only when ALL sub-lanes
  // report ready for that input.
  // ---------------------------------------------------------------
  always_comb begin : aggregate_in_ready
    integer iter_var0;
    integer iter_var1;
    for (iter_var0 = 0; iter_var0 < NUM_IN; iter_var0 = iter_var0 + 1) begin : air_in_loop
      in_ready[iter_var0] = 1'b1;
      for (iter_var1 = 0; iter_var1 < NUM_LANES; iter_var1 = iter_var1 + 1) begin : air_lane_loop
        if (!lane_in_ready[iter_var1][iter_var0]) begin : air_not_ready
          in_ready[iter_var0] = 1'b0;
        end : air_not_ready
      end : air_lane_loop
    end : air_in_loop
  end : aggregate_in_ready

endmodule : fabric_spatial_sw_decomp
