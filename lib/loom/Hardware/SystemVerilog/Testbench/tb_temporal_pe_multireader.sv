//===-- tb_temporal_pe_multireader.sv - Multi-reader register test -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Tests multi-reader register identity tracking: two instructions share a
// register read. The register FIFO entry must NOT be dequeued until both
// distinct instructions have fired (identity-tracked consume).
//
// Config: NUM_REGISTERS=2, NUM_INSTRUCTIONS=3, REG_FIFO_DEPTH=4
//   insn 0 (tag=1): op[0]=input, op[1]=input, result -> reg 0 (write)
//   insn 1 (tag=2): op[0]=reg 0 (read), op[1]=input, result -> output
//   insn 2 (tag=3): op[0]=reg 0 (read), op[1]=input, result -> output
// insn 1 and insn 2 both read reg 0 -> reg_reader_mask[0] = 3'b110
// After insn 0 fires and writes reg 0, insn 1 firing alone must NOT dequeue
// the register entry. Only after insn 2 also fires should dequeue occur.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_temporal_pe_multireader;
  localparam int NUM_INPUTS           = 2;
  localparam int NUM_OUTPUTS          = 1;
  localparam int DATA_WIDTH           = 32;
  localparam int TAG_WIDTH            = 4;
  localparam int NUM_FU_TYPES         = 1;
  localparam int NUM_REGISTERS        = 2;
  localparam int NUM_INSTRUCTIONS     = 3;
  localparam int REG_FIFO_DEPTH       = 4;
  localparam int SHARED_OPERAND_BUFFER         = 0;
  localparam int OPERAND_BUFFER_SIZE  = 0;

  localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH;
  localparam int SAFE_PW = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1;

  // Derived config layout params (must match fabric_temporal_pe.sv)
  localparam int REG_BITS     = 1 + $clog2(NUM_REGISTERS > 1 ? NUM_REGISTERS : 2);
  localparam int FU_SEL_BITS  = (NUM_FU_TYPES > 1) ? $clog2(NUM_FU_TYPES) : 0;
  localparam int RES_BITS     = 1 + $clog2(NUM_REGISTERS > 1 ? NUM_REGISTERS : 2);
  localparam int RESULT_WIDTH = RES_BITS + TAG_WIDTH;
  localparam int INSN_WIDTH   = 1 + TAG_WIDTH + FU_SEL_BITS + NUM_INPUTS * REG_BITS + NUM_OUTPUTS * RESULT_WIDTH;
  localparam int INSN_VALID_LSB = 0;
  localparam int INSN_TAG_LSB = INSN_VALID_LSB + 1;
  localparam int INSN_OPERANDS_LSB = INSN_TAG_LSB + TAG_WIDTH + FU_SEL_BITS;
  localparam int INSN_RESULTS_LSB = INSN_OPERANDS_LSB + NUM_INPUTS * REG_BITS;

  logic clk, rst_n;
  logic [NUM_INPUTS-1:0]                 in_valid;
  logic [NUM_INPUTS-1:0]                 in_ready;
  logic [NUM_INPUTS-1:0][SAFE_PW-1:0]   in_data;
  logic [NUM_OUTPUTS-1:0]               out_valid;
  logic [NUM_OUTPUTS-1:0]               out_ready;
  logic [NUM_OUTPUTS-1:0][SAFE_PW-1:0]  out_data;
  logic [255:0] cfg_data;
  logic        error_valid;
  logic [15:0] error_code;

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
    .clk(clk), .rst_n(rst_n),
    .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data),
    .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
    .cfg_data(cfg_data[dut.CONFIG_WIDTH > 0 ? dut.CONFIG_WIDTH-1 : 0 : 0]),
    .error_valid(error_valid), .error_code(error_code)
  );

  initial begin : clk_gen
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin : test
    integer pass_count;
    integer iter_var0;
    integer fire_count;
    pass_count = 0;
    rst_n = 0;
    in_valid = '0;
    out_ready = '1;
    in_data = '0;
    cfg_data = '0;

    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // ---------------------------------------------------------------
    // Build instruction config
    // Per-insn layout (LSB -> MSB):
    //   valid | tag | fu_sel | operand fields | result fields
    // Operand field format: reg_idx (LSB) + is_reg (MSB)
    // Result field format:  res_tag (LSB) + reg_idx + is_reg (MSB)
    // ---------------------------------------------------------------
    cfg_data = '0;

    // Insn 0 (tag=1): op[0]=input, op[1]=input, result -> reg 0 (write)
    begin : cfg_insn0
      automatic int base = 0 * INSN_WIDTH;
      automatic int op0_base = base + INSN_OPERANDS_LSB + 0 * REG_BITS;
      automatic int op1_base = base + INSN_OPERANDS_LSB + 1 * REG_BITS;
      automatic int res0_base = base + INSN_RESULTS_LSB + 0 * RESULT_WIDTH;
      // Result: res_tag=0 (must be 0 for register writes), res_reg_idx=0, res_is_reg=1
      cfg_data[res0_base +: TAG_WIDTH] = TAG_WIDTH'(0); // res_tag (must be 0 for reg write)
      cfg_data[res0_base + TAG_WIDTH] = 1'b0;           // res_reg_idx = 0
      cfg_data[res0_base + RESULT_WIDTH - 1] = 1'b1;    // res_is_reg = 1
      // Op[0]: input (is_reg=0)
      cfg_data[op0_base + REG_BITS - 1] = 1'b0;
      // Op[1]: input (is_reg=0)
      cfg_data[op1_base + REG_BITS - 1] = 1'b0;
      // Tag = 1
      cfg_data[base + INSN_TAG_LSB +: TAG_WIDTH] = TAG_WIDTH'(1);
      // Valid
      cfg_data[base + INSN_VALID_LSB] = 1'b1;
    end

    // Insn 1 (tag=2): op[0]=reg 0 (read), op[1]=input, result -> output
    begin : cfg_insn1
      automatic int base = 1 * INSN_WIDTH;
      automatic int op0_base = base + INSN_OPERANDS_LSB + 0 * REG_BITS;
      automatic int op1_base = base + INSN_OPERANDS_LSB + 1 * REG_BITS;
      automatic int res0_base = base + INSN_RESULTS_LSB + 0 * RESULT_WIDTH;
      // Result: res_tag=2, res_is_reg=0
      cfg_data[res0_base +: TAG_WIDTH] = TAG_WIDTH'(2);
      cfg_data[res0_base + RESULT_WIDTH - 1] = 1'b0;    // res_is_reg = 0
      // Op[0]: reg 0 (is_reg=1, reg_idx=0)
      cfg_data[op0_base +: (REG_BITS - 1)] = '0;         // reg_idx = 0
      cfg_data[op0_base + REG_BITS - 1] = 1'b1;          // is_reg = 1
      // Op[1]: input (is_reg=0)
      cfg_data[op1_base + REG_BITS - 1] = 1'b0;
      // Tag = 2
      cfg_data[base + INSN_TAG_LSB +: TAG_WIDTH] = TAG_WIDTH'(2);
      // Valid
      cfg_data[base + INSN_VALID_LSB] = 1'b1;
    end

    // Insn 2 (tag=3): op[0]=reg 0 (read), op[1]=input, result -> output
    begin : cfg_insn2
      automatic int base = 2 * INSN_WIDTH;
      automatic int op0_base = base + INSN_OPERANDS_LSB + 0 * REG_BITS;
      automatic int op1_base = base + INSN_OPERANDS_LSB + 1 * REG_BITS;
      automatic int res0_base = base + INSN_RESULTS_LSB + 0 * RESULT_WIDTH;
      // Result: res_tag=3, res_is_reg=0
      cfg_data[res0_base +: TAG_WIDTH] = TAG_WIDTH'(3);
      cfg_data[res0_base + RESULT_WIDTH - 1] = 1'b0;    // res_is_reg = 0
      // Op[0]: reg 0 (is_reg=1, reg_idx=0)
      cfg_data[op0_base +: (REG_BITS - 1)] = '0;         // reg_idx = 0
      cfg_data[op0_base + REG_BITS - 1] = 1'b1;          // is_reg = 1
      // Op[1]: input (is_reg=0)
      cfg_data[op1_base + REG_BITS - 1] = 1'b0;
      // Tag = 3
      cfg_data[base + INSN_TAG_LSB +: TAG_WIDTH] = TAG_WIDTH'(3);
      // Valid
      cfg_data[base + INSN_VALID_LSB] = 1'b1;
    end
    @(posedge clk);

    // Check 1: no config error
    if (error_valid !== 1'b0) begin : check_cfg
      $fatal(1, "unexpected config error: code=%0d", error_code);
    end
    pass_count = pass_count + 1;

    // Step 1: Fire insn 0 (tag=1) to write register 0.
    // insn 0: op[0]=input, op[1]=input, result -> reg 0 (write)
    in_data[0] = {TAG_WIDTH'(1), 32'h0000_CAFE};
    in_data[1] = {TAG_WIDTH'(1), 32'h0000_0001};
    in_valid = 2'b11;
    @(posedge clk);
    in_valid = '0;

    // Wait for insn 0 to fire and write register 0.
    // Fire is combinational and self-clears. Use FIFO state as indicator.
    repeat (4) @(posedge clk);

    // Check 2: no error after register write
    if (error_valid !== 1'b0) begin : check_reg_write
      $fatal(1, "error after register write: code=%0d", error_code);
    end
    // Verify register 0 FIFO has one entry (written by insn 0)
    if (dut.reg_fifo_cnt[0] !== (dut.RFIFO_IDX_W+1)'(1)) begin : check_reg0_cnt
      $fatal(1, "reg 0 FIFO should have 1 entry, got %0d", dut.reg_fifo_cnt[0]);
    end
    pass_count = pass_count + 1;

    // Step 2: Provide operands for insn 1 (tag=2).
    // insn 1: op[0]=reg 0 (read), op[1]=input, result -> output
    // op[0] is register-sourced: op_valid comes from reg_fifo_empty, not buffer.
    // Only op[1] needs to be provided via input stream.
    // Since insn 1 reads from port 1 (op[1]=input), send tag=2 on port 1.
    // Port 0 is register-sourced so any input on port 0 with tag=2 will
    // be accepted but overridden by the register value in op_valid merge.
    in_data[0] = {TAG_WIDTH'(2), 32'h0000_0002};
    in_data[1] = {TAG_WIDTH'(2), 32'h0000_0003};
    in_valid = 2'b11;
    @(posedge clk);
    in_valid = '0;

    // Wait for insn 1 to fire. Check buffer/reg state.
    repeat (4) @(posedge clk);

    // Check 3: insn 1's buffer should be cleared (it fired).
    // Operand buffer for insn 1 input-sourced ops should be empty.
    if (dut.op_buf_valid[1] != '0) begin : check_insn1_buf
      $fatal(1, "insn 1 buffer not cleared (op_buf_valid=%b), insn 1 may not have fired",
             dut.op_buf_valid[1]);
    end
    // After insn 1 fires alone, reg 0 should NOT be dequeued (identity tracking).
    // reg_rd_consumed[0] should have bit 1 set (insn 1 consumed), but bit 2
    // not set (insn 2 not yet consumed). FIFO count should still be 1.
    if (dut.reg_fifo_cnt[0] !== (dut.RFIFO_IDX_W+1)'(1)) begin : check_reg0_retain
      $fatal(1, "reg 0 FIFO prematurely dequeued after insn 1 (count=%0d, expected 1)",
             dut.reg_fifo_cnt[0]);
    end
    pass_count = pass_count + 1;

    // Step 3: Provide operands for insn 2 (tag=3).
    // insn 2: op[0]=reg 0 (read), op[1]=input, result -> output
    in_data[0] = {TAG_WIDTH'(3), 32'h0000_0004};
    in_data[1] = {TAG_WIDTH'(3), 32'h0000_0005};
    in_valid = 2'b11;
    @(posedge clk);
    in_valid = '0;

    // Wait for insn 2 to fire.
    repeat (4) @(posedge clk);

    // Check 4: insn 2 buffer cleared AND reg 0 now dequeued (all readers done)
    if (dut.op_buf_valid[2] != '0) begin : check_insn2_buf
      $fatal(1, "insn 2 buffer not cleared (op_buf_valid=%b), insn 2 may not have fired",
             dut.op_buf_valid[2]);
    end
    if (dut.reg_fifo_cnt[0] !== '0) begin : check_reg0_dequeue
      $fatal(1, "reg 0 FIFO not dequeued after both readers consumed (count=%0d)",
             dut.reg_fifo_cnt[0]);
    end
    pass_count = pass_count + 1;

    // Check 5: no errors throughout first sequence
    if (error_valid !== 1'b0) begin : check_final
      $fatal(1, "unexpected error at end: code=%0d", error_code);
    end
    pass_count = pass_count + 1;

    // ---------------------------------------------------------------
    // Repeated-same-reader stress: write reg 0 twice, then fire
    // insn 1 twice (both values) WITHOUT insn 2 firing. After insn 1
    // consumes both entries, neither entry should be dequeued because
    // insn 2 (the other reader) hasn't consumed either. This
    // distinguishes identity-tracked dequeue from a simple scalar
    // counter.
    // ---------------------------------------------------------------

    // Reset to clear all state from the first sequence.
    rst_n = 0;
    in_valid = '0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    // Re-apply config (cfg_data is still driven).
    @(posedge clk);

    // Write reg 0 twice via insn 0 (tag=1).
    // First write: op[0]=0xA, op[1]=0x1
    in_data[0] = {TAG_WIDTH'(1), 32'h0000_000A};
    in_data[1] = {TAG_WIDTH'(1), 32'h0000_0001};
    in_valid = 2'b11;
    @(posedge clk);
    in_valid = '0;
    repeat (4) @(posedge clk);

    // Second write: op[0]=0xB, op[1]=0x2
    in_data[0] = {TAG_WIDTH'(1), 32'h0000_000B};
    in_data[1] = {TAG_WIDTH'(1), 32'h0000_0002};
    in_valid = 2'b11;
    @(posedge clk);
    in_valid = '0;
    repeat (4) @(posedge clk);

    // Check 6: reg 0 FIFO should have 2 entries after two writes.
    if (dut.reg_fifo_cnt[0] !== (dut.RFIFO_IDX_W+1)'(2)) begin : check_reg0_two
      $fatal(1, "reg 0 FIFO should have 2 entries after two writes, got %0d",
             dut.reg_fifo_cnt[0]);
    end
    pass_count = pass_count + 1;

    // Fire insn 1 (tag=2) to consume first reg 0 entry.
    in_data[0] = {TAG_WIDTH'(2), 32'h0000_0010};
    in_data[1] = {TAG_WIDTH'(2), 32'h0000_0020};
    in_valid = 2'b11;
    @(posedge clk);
    in_valid = '0;
    repeat (4) @(posedge clk);

    // Check 7: after insn 1 fires once, reg 0 should still have 2 entries.
    // Identity tracking: insn 2 hasn't consumed, so no dequeue allowed.
    if (dut.reg_fifo_cnt[0] !== (dut.RFIFO_IDX_W+1)'(2)) begin : check_reg0_retain2
      $fatal(1, "reg 0 FIFO should still have 2 entries (insn 2 not consumed), got %0d",
             dut.reg_fifo_cnt[0]);
    end
    pass_count = pass_count + 1;

    // Fire insn 1 (tag=2) again to consume second reg 0 entry.
    in_data[0] = {TAG_WIDTH'(2), 32'h0000_0030};
    in_data[1] = {TAG_WIDTH'(2), 32'h0000_0040};
    in_valid = 2'b11;
    @(posedge clk);
    in_valid = '0;
    repeat (4) @(posedge clk);

    // Check 8: after insn 1 fires twice (consuming both values), reg 0
    // should still have 2 entries. Neither can be dequeued because insn 2
    // hasn't consumed either entry yet.
    if (dut.reg_fifo_cnt[0] !== (dut.RFIFO_IDX_W+1)'(2)) begin : check_reg0_no_deq
      $fatal(1, "reg 0 FIFO should still have 2 entries (insn 2 never fired), got %0d",
             dut.reg_fifo_cnt[0]);
    end
    pass_count = pass_count + 1;

    $display("PASS: tb_temporal_pe_multireader (%0d checks)", pass_count);
    $finish;
  end

  initial begin : timeout
    #40000;
    $fatal(1, "TIMEOUT");
  end
endmodule
