// fabric_temporal_pe_fu_slot.sv -- Per-FU slot wrapper (Layer 2).
//
// Wraps one function_unit instance with:
//
//   1. Latency pipeline: shift register of depth (CONFIGURED_LATENCY -
//      INTRINSIC_LATENCY) stages.  The FU body provides intrinsic latency
//      compute; this wrapper adds the remaining retiming stages.
//      When CONFIGURED_LATENCY == INTRINSIC_LATENCY (or both 0), no
//      extra pipeline stages are added.
//
//   2. Interval throttle: countdown counter that blocks re-fire until
//      INTERVAL cycles have elapsed since the last fire.  When INTERVAL
//      == 1, the FU is fully pipelined and no throttle is needed.
//
//   3. Busy logic: The FU is busy when:
//      - Output registers (owned by output_arb) are undrained, OR
//      - Interval counter is nonzero, OR
//      - There are results inflight in the latency pipeline
//
//   For dataflow FUs (CONFIGURED_LATENCY == -1, INTERVAL == -1), the
//   pipeline and throttle are bypassed; the FU body manages its own timing.
//
// The FU body itself is NOT instantiated here -- it is connected via
// the fu_in/fu_out ports.  The top-level PE wires the actual FU body
// to these ports.  This module only handles the latency pipeline and
// interval throttle between the FU body outputs and the FU-local
// output registers.

module fabric_temporal_pe_fu_slot
  import fabric_pkg::*;
#(
  parameter int unsigned NUM_FU_IN           = 2,
  parameter int unsigned NUM_FU_OUT          = 2,
  parameter int unsigned DATA_WIDTH          = 32,
  parameter int unsigned TAG_WIDTH           = 4,
  // Configured latency from fabric MLIR (fire to completion).
  // -1 means dataflow state-machine FU (no pipeline, no throttle).
  parameter int signed   CONFIGURED_LATENCY  = 0,
  // Intrinsic latency of the FU body (combinational = 0, pipelined > 0).
  parameter int unsigned INTRINSIC_LATENCY   = 0,
  // Configured interval (minimum cycles between fires).
  // -1 means dataflow state-machine FU.
  parameter int signed   CONFIGURED_INTERVAL = 1
)(
  input  logic        clk,
  input  logic        rst_n,

  // --- Fire strobe from scheduler ---
  input  logic        fire,

  // --- FU body output (after intrinsic compute) ---
  input  logic [NUM_FU_OUT-1:0]    fu_out_valid,
  input  logic [DATA_WIDTH-1:0]    fu_out_data   [NUM_FU_OUT],

  // --- Pipeline output (to output registers in output_arb) ---
  output logic [NUM_FU_OUT-1:0]    pipe_out_valid,
  output logic [DATA_WIDTH-1:0]    pipe_out_data  [NUM_FU_OUT],

  // --- Output register drain status (from output_arb) ---
  // Asserted per output when the FU-local output register still holds
  // an undrained valid result.
  input  logic [NUM_FU_OUT-1:0]    out_reg_occupied,

  // --- Busy status ---
  output logic                     busy
);

  // ---------------------------------------------------------------
  // Derived parameters
  // ---------------------------------------------------------------
  // For dataflow FUs, no pipeline or throttle
  localparam bit IS_DATAFLOW = (CONFIGURED_LATENCY < 0);

  // Extra pipeline depth beyond intrinsic
  localparam int unsigned EXTRA_LATENCY =
    IS_DATAFLOW ? 0 :
    (CONFIGURED_LATENCY > INTRINSIC_LATENCY) ?
      (CONFIGURED_LATENCY - INTRINSIC_LATENCY) : 0;

  // Interval counter width
  localparam int unsigned INTERVAL_VAL =
    IS_DATAFLOW ? 0 :
    (CONFIGURED_INTERVAL > 1) ? (CONFIGURED_INTERVAL - 1) : 0;
  localparam int unsigned INTERVAL_CNT_W =
    (INTERVAL_VAL > 0) ? $clog2(INTERVAL_VAL + 1) : 1;

  // ---------------------------------------------------------------
  // Dataflow FU: direct passthrough, no pipeline/throttle
  // ---------------------------------------------------------------
  generate
    if (IS_DATAFLOW) begin : gen_dataflow

      // Direct passthrough
      assign pipe_out_valid = fu_out_valid;
      always_comb begin : passthrough_data
        integer iter_var0;
        for (iter_var0 = 0; iter_var0 < NUM_FU_OUT; iter_var0 = iter_var0 + 1) begin : pass_out
          pipe_out_data[iter_var0] = fu_out_data[iter_var0];
        end : pass_out
      end : passthrough_data

      // Dataflow FU busy: only when output regs occupied
      assign busy = |out_reg_occupied;

    end : gen_dataflow

    // ---------------------------------------------------------------
    // Normal FU: latency pipeline + interval throttle
    // ---------------------------------------------------------------
    else begin : gen_normal

      // ---------------------------------------------------------------
      // Interval throttle
      // ---------------------------------------------------------------
      logic [INTERVAL_CNT_W-1:0] interval_cnt;
      logic interval_busy;

      if (INTERVAL_VAL > 0) begin : gen_interval

        always_ff @(posedge clk or negedge rst_n) begin : interval_seq
          if (!rst_n) begin : interval_reset
            interval_cnt <= '0;
          end : interval_reset
          else begin : interval_op
            if (fire) begin : interval_reload
              interval_cnt <= INTERVAL_CNT_W'(INTERVAL_VAL);
            end : interval_reload
            else if (interval_cnt != '0) begin : interval_dec
              interval_cnt <= interval_cnt - 1'b1;
            end : interval_dec
          end : interval_op
        end : interval_seq

        assign interval_busy = (interval_cnt != '0);

      end : gen_interval
      else begin : gen_no_interval

        assign interval_cnt  = '0;
        assign interval_busy = 1'b0;

      end : gen_no_interval

      // ---------------------------------------------------------------
      // Latency pipeline (shift register)
      // ---------------------------------------------------------------
      logic pipeline_busy;

      if (EXTRA_LATENCY == 0) begin : gen_no_pipe

          // No extra latency: FU output goes directly to output stage
          assign pipe_out_valid = fu_out_valid;
          always_comb begin : no_pipe_data
            integer iter_var0;
            for (iter_var0 = 0; iter_var0 < NUM_FU_OUT; iter_var0 = iter_var0 + 1) begin : no_pipe_out
              pipe_out_data[iter_var0] = fu_out_data[iter_var0];
            end : no_pipe_out
          end : no_pipe_data
          assign pipeline_busy = 1'b0;

        end : gen_no_pipe
        else begin : gen_pipe

          // Pipeline stages: EXTRA_LATENCY deep, NUM_FU_OUT wide
          // Stage 0 = FU output, stage EXTRA_LATENCY = pipe output
          logic [NUM_FU_OUT-1:0]    stage_valid [0:EXTRA_LATENCY-1];
          logic [DATA_WIDTH-1:0]    stage_data  [0:EXTRA_LATENCY-1][0:NUM_FU_OUT-1];

          always_ff @(posedge clk or negedge rst_n) begin : pipe_seq
            integer iter_var0;
            integer iter_var1;
            if (!rst_n) begin : pipe_reset
              for (iter_var0 = 0; iter_var0 < EXTRA_LATENCY; iter_var0 = iter_var0 + 1) begin : rst_stage
                stage_valid[iter_var0] <= '0;
                for (iter_var1 = 0; iter_var1 < NUM_FU_OUT; iter_var1 = iter_var1 + 1) begin : rst_data
                  stage_data[iter_var0][iter_var1] <= '0;
                end : rst_data
              end : rst_stage
            end : pipe_reset
            else begin : pipe_op
              // Stage 0 takes FU output
              stage_valid[0] <= fu_out_valid;
              for (iter_var1 = 0; iter_var1 < NUM_FU_OUT; iter_var1 = iter_var1 + 1) begin : s0_data
                stage_data[0][iter_var1] <= fu_out_data[iter_var1];
              end : s0_data

              // Shift remaining stages
              for (iter_var0 = 1; iter_var0 < EXTRA_LATENCY; iter_var0 = iter_var0 + 1) begin : shift_stage
                stage_valid[iter_var0] <= stage_valid[iter_var0 - 1];
                for (iter_var1 = 0; iter_var1 < NUM_FU_OUT; iter_var1 = iter_var1 + 1) begin : shift_data
                  stage_data[iter_var0][iter_var1] <= stage_data[iter_var0 - 1][iter_var1];
                end : shift_data
              end : shift_stage
            end : pipe_op
          end : pipe_seq

          // Output from the last pipeline stage
          assign pipe_out_valid = stage_valid[EXTRA_LATENCY - 1];
          always_comb begin : pipe_output
            integer iter_var0;
            for (iter_var0 = 0; iter_var0 < NUM_FU_OUT; iter_var0 = iter_var0 + 1) begin : pipe_out
              pipe_out_data[iter_var0] = stage_data[EXTRA_LATENCY - 1][iter_var0];
            end : pipe_out
          end : pipe_output

          // Pipeline busy: any stage has valid data
          always_comb begin : pipe_busy_check
            integer iter_var0;
            pipeline_busy = 1'b0;
            for (iter_var0 = 0; iter_var0 < EXTRA_LATENCY; iter_var0 = iter_var0 + 1) begin : chk_stage
              if (|stage_valid[iter_var0]) begin : stage_active
                pipeline_busy = 1'b1;
              end : stage_active
            end : chk_stage
          end : pipe_busy_check

        end : gen_pipe

      // ---------------------------------------------------------------
      // Busy: union of all blocking conditions
      // ---------------------------------------------------------------
      assign busy = (|out_reg_occupied) | interval_busy | pipeline_busy;

    end : gen_normal
  endgenerate

endmodule : fabric_temporal_pe_fu_slot
