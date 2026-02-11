//===-- tb_temporal_pe_stress.sv - Temporal PE mixed stress test -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Long-run stress test with mixed tags and randomized ready/valid patterns.
// Uses deterministic LFSR stimulus (no nondeterministic random functions).
//
// Coverage intent:
// - Mixed tag arrivals across both inputs.
// - Backpressure on output via randomized out_ready.
// - Long-run operand buffering/firing stability.
//
// Scoreboard:
// - Models per-instruction operand-buffer occupancy.
// - Mirrors issue/commit timing (fu_launch/fu_busy) to predict fire cycles.
// - Checks DUT fire behavior, output handshake/tag, and buffer state every cycle.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_temporal_pe_stress;
  localparam int NUM_INPUTS           = 2;
  localparam int NUM_OUTPUTS          = 1;
  localparam int DATA_WIDTH           = 32;
  localparam int TAG_WIDTH            = 4;
  localparam int NUM_FU_TYPES         = 1;
  localparam int NUM_REGISTERS        = 0;
  localparam int NUM_INSTRUCTIONS     = 2;
  localparam int REG_FIFO_DEPTH       = 0;
  localparam int SHARED_OPERAND_BUFFER         = 0;
  localparam int OPERAND_BUFFER_SIZE  = 0;

  localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH;
  localparam int SAFE_PW = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1;
  localparam int INSN_WIDTH = 1 + TAG_WIDTH + NUM_OUTPUTS * TAG_WIDTH;
  localparam int INSN_IDX_W = $clog2(NUM_INSTRUCTIONS > 1 ? NUM_INSTRUCTIONS : 2);
  localparam int INSN_VALID_LSB = 0;
  localparam int INSN_TAG_LSB = INSN_VALID_LSB + 1;
  localparam int INSN_RESULTS_LSB = INSN_TAG_LSB + TAG_WIDTH;

  localparam logic [TAG_WIDTH-1:0] TAG0 = TAG_WIDTH'(1);
  localparam logic [TAG_WIDTH-1:0] TAG1 = TAG_WIDTH'(2);

  logic clk;
  logic rst_n;

  logic [NUM_INPUTS-1:0]               in_valid;
  logic [NUM_INPUTS-1:0]               in_ready;
  logic [NUM_INPUTS-1:0][SAFE_PW-1:0]  in_data;

  logic [NUM_OUTPUTS-1:0]              out_valid;
  logic [NUM_OUTPUTS-1:0]              out_ready;
  logic [NUM_OUTPUTS-1:0][SAFE_PW-1:0] out_data;

  logic [255:0] cfg_data;
  logic         error_valid;
  logic [15:0]  error_code;

  fabric_temporal_pe #(
    .NUM_INPUTS(NUM_INPUTS),
    .NUM_OUTPUTS(NUM_OUTPUTS),
    .DATA_WIDTH(DATA_WIDTH),
    .TAG_WIDTH(TAG_WIDTH),
    .NUM_FU_TYPES(NUM_FU_TYPES),
    .NUM_REGISTERS(NUM_REGISTERS),
    .NUM_INSTRUCTIONS(NUM_INSTRUCTIONS),
    .REG_FIFO_DEPTH(REG_FIFO_DEPTH),
    .SHARED_OPERAND_BUFFER(SHARED_OPERAND_BUFFER),
    .OPERAND_BUFFER_SIZE(OPERAND_BUFFER_SIZE)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .in_valid(in_valid),
    .in_ready(in_ready),
    .in_data(in_data),
    .out_valid(out_valid),
    .out_ready(out_ready),
    .out_data(out_data),
    .cfg_data(cfg_data[dut.CONFIG_WIDTH > 0 ? dut.CONFIG_WIDTH-1 : 0 : 0]),
    .error_valid(error_valid),
    .error_code(error_code)
  );

  function automatic logic [31:0] lfsr_next(input logic [31:0] state);
    logic feedback;
    begin : fn
      feedback = state[31] ^ state[21] ^ state[1] ^ state[0];
      lfsr_next = {state[30:0], feedback};
    end
  endfunction

  function automatic int tag_to_slot(input logic [TAG_WIDTH-1:0] tag);
    begin : fn
      tag_to_slot = (tag == TAG0) ? 0 : 1;
    end
  endfunction

  initial begin : clk_gen
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  initial begin : test
    integer pass_count;
    integer cycle_count;
    integer fire_count_model;
    integer fire_count_dut;
    integer value_counter;
    integer iter_var0;
    integer iter_var1;
    integer slot_idx;

    logic [31:0] lfsr;
    logic model_fire;
    logic [INSN_IDX_W-1:0] model_matched_slot;
    logic [INSN_IDX_W-1:0] model_issued_slot;
    logic [INSN_IDX_W-1:0] model_commit_slot;
    logic model_insn_fire_ready;
    logic model_fu_busy;
    logic model_fu_launch;
    logic [TAG_WIDTH-1:0] model_out_tag;

    logic [NUM_INPUTS-1:0] offered_valid;
    logic [NUM_INPUTS-1:0][TAG_WIDTH-1:0] offered_tag;
    logic [NUM_INPUTS-1:0][DATA_WIDTH-1:0] offered_value;
    logic [NUM_INPUTS-1:0] accepted;

    // [instruction][input]
    logic [NUM_INSTRUCTIONS-1:0][NUM_INPUTS-1:0] model_buf_valid;

    pass_count = 0;
    cycle_count = 0;
    fire_count_model = 0;
    fire_count_dut = 0;
    value_counter = 0;

    rst_n = 1'b0;
    in_valid = '0;
    in_data = '0;
    out_ready = '0;
    cfg_data = '0;

    offered_valid = '0;
    offered_tag = '0;
    offered_value = '0;
    model_buf_valid = '0;
    accepted = '0;
    model_matched_slot = '0;
    model_issued_slot = '0;
    model_commit_slot = '0;
    model_insn_fire_ready = 1'b0;
    model_fu_busy = 1'b0;
    model_fu_launch = 1'b0;
    model_out_tag = '0;

    // Configure two valid instructions.
    // insn 0: tag=1, out_tag=1
    cfg_data[0 * INSN_WIDTH + INSN_VALID_LSB] = 1'b1;
    cfg_data[0 * INSN_WIDTH + INSN_TAG_LSB +: TAG_WIDTH] = TAG0;
    cfg_data[0 * INSN_WIDTH + INSN_RESULTS_LSB +: TAG_WIDTH] = TAG0;
    // insn 1: tag=2, out_tag=2
    cfg_data[1 * INSN_WIDTH + INSN_VALID_LSB] = 1'b1;
    cfg_data[1 * INSN_WIDTH + INSN_TAG_LSB +: TAG_WIDTH] = TAG1;
    cfg_data[1 * INSN_WIDTH + INSN_RESULTS_LSB +: TAG_WIDTH] = TAG1;

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    if (error_valid !== 1'b0) begin : check_reset
      $fatal(1, "error_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    lfsr = 32'h1ACE_B00C;

    // 400-cycle deterministic stress run.
    for (iter_var0 = 0; iter_var0 < 400; iter_var0 = iter_var0 + 1) begin : stress_loop
      @(negedge clk);

      // Output backpressure jitter.
      out_ready[0] = lfsr[0] | lfsr[5];

      // Drive deterministic random offers. Offers are one-cycle pulses;
      // unaccepted offers are dropped so traffic remains live.
      for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1 = iter_var1 + 1) begin : drive_in
        offered_valid[iter_var1] = lfsr[1 + iter_var1] | lfsr[9 + iter_var1];
        offered_tag[iter_var1] = lfsr[17 + iter_var1] ? TAG1 : TAG0;
        offered_value[iter_var1] = DATA_WIDTH'(value_counter);
        if (offered_valid[iter_var1]) begin : next_value
          value_counter = value_counter + 1;
        end
        in_valid[iter_var1] = offered_valid[iter_var1];
        in_data[iter_var1] = {offered_tag[iter_var1], offered_value[iter_var1]};
      end

      @(posedge clk);

      if (error_valid) begin : check_no_error
        $fatal(1, "unexpected error during stress: code=%0d at cycle=%0d", error_code, iter_var0);
      end

      // Predict fire using pre-update model state.
      model_insn_fire_ready = 1'b0;
      model_matched_slot = '0;
      for (slot_idx = 0; slot_idx < NUM_INSTRUCTIONS; slot_idx = slot_idx + 1) begin : find_ready
        if (!model_insn_fire_ready &&
            model_buf_valid[slot_idx][0] &&
            model_buf_valid[slot_idx][1]) begin : pick
          model_insn_fire_ready = 1'b1;
          model_matched_slot = INSN_IDX_W'(slot_idx);
        end
      end
      model_fu_launch = model_insn_fire_ready && !model_fu_busy;
      model_commit_slot = model_fu_launch ? model_matched_slot : model_issued_slot;
      model_fire = model_insn_fire_ready && out_ready[0] && (model_fu_busy || model_fu_launch);
      model_out_tag = (model_commit_slot == INSN_IDX_W'(0)) ? TAG0 : TAG1;

      if (dut.fire !== model_fire) begin : check_fire
        $fatal(1,
               "fire mismatch at cycle=%0d: dut.fire=%0b model_fire=%0b model_buf0=%b model_buf1=%b out_ready=%0b model_busy=%0b model_launch=%0b model_match=%0d model_issue=%0d",
               iter_var0, dut.fire, model_fire, model_buf_valid[0], model_buf_valid[1], out_ready[0],
               model_fu_busy, model_fu_launch, model_matched_slot, model_issued_slot);
      end

      if (out_valid[0] !== model_fire) begin : check_out_valid
        $fatal(1,
               "out_valid mismatch at cycle=%0d: out_valid=%0b model_fire=%0b",
               iter_var0, out_valid[0], model_fire);
      end

      if (model_fire) begin : check_out_tag
        if (out_data[0][DATA_WIDTH +: TAG_WIDTH] !== model_out_tag) begin : bad_tag
          $fatal(1,
                 "out tag mismatch at cycle=%0d: out_tag=%0h expected=%0h commit_slot=%0d",
                 iter_var0, out_data[0][DATA_WIDTH +: TAG_WIDTH], model_out_tag, model_commit_slot);
        end
      end

      if (dut.fire) begin : count_dut_fire
        fire_count_dut = fire_count_dut + 1;
      end
      if (model_fire) begin : count_model_fire
        fire_count_model = fire_count_model + 1;
      end

      // Handshake outcomes for this cycle.
      for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1 = iter_var1 + 1) begin : cap_acc
        accepted[iter_var1] = in_valid[iter_var1] && in_ready[iter_var1];
      end

      // Sequential model update: clear fired slot, then accept new operands.
      if (model_fire) begin : clr_fired
        model_buf_valid[model_commit_slot][0] = 1'b0;
        model_buf_valid[model_commit_slot][1] = 1'b0;
      end

      for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1 = iter_var1 + 1) begin : upd_acc
        if (accepted[iter_var1]) begin : do_acc
          slot_idx = tag_to_slot(offered_tag[iter_var1]);
          model_buf_valid[slot_idx][iter_var1] = 1'b1;
        end
      end

      // Model FU issue/commit state.
      if (model_fu_launch && model_fire) begin : model_launch_and_commit
        model_fu_busy = 1'b0;
        model_issued_slot = model_matched_slot;
      end else if (model_fu_launch) begin : model_launch_only
        model_fu_busy = 1'b1;
        model_issued_slot = model_matched_slot;
      end else if (model_fire) begin : model_commit_only
        model_fu_busy = 1'b0;
      end

      // Allow DUT sequential state to settle, then compare operand buffer state.
      #1;
      if (dut.op_buf_valid !== model_buf_valid) begin : check_buf_state
        $fatal(1,
               "op_buf_valid mismatch at cycle=%0d: dut0=%b dut1=%b model0=%b model1=%b model_commit=%0d model_issue=%0d model_match=%0d",
               iter_var0, dut.op_buf_valid[0], dut.op_buf_valid[1], model_buf_valid[0], model_buf_valid[1],
               model_commit_slot, model_issued_slot, model_matched_slot);
      end

      lfsr = lfsr_next(lfsr);
      cycle_count = cycle_count + 1;
    end

    // Final checks.
    if (fire_count_dut !== fire_count_model) begin : check_fire_count
      $fatal(1, "fire count mismatch: dut=%0d model=%0d", fire_count_dut, fire_count_model);
    end
    if (fire_count_dut < 20) begin : check_activity
      $fatal(1, "stress activity too low: fire_count=%0d", fire_count_dut);
    end
    pass_count = pass_count + 1;

    if (error_valid !== 1'b0) begin : check_final_error
      $fatal(1, "unexpected final error: code=%0d", error_code);
    end
    pass_count = pass_count + 1;

    $display("PASS: tb_temporal_pe_stress cycles=%0d fires=%0d (%0d checks)",
             cycle_count, fire_count_dut, pass_count);
    $finish;
  end

  initial begin : timeout
    #60000;
    $fatal(1, "TIMEOUT");
  end

endmodule
