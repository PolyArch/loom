//===-- tb_temporal_pe_shared_buf.sv - Shared-buf collision regr -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Tests shared operand buffer collision prevention: concurrent
// different-tag arrivals must not merge into the same buffer entry.
// With OPERAND_BUFFER_SIZE=2, two instructions (tag=1, tag=2) receive
// interleaved operands. Both must fire independently without error.
//
// Checks:
//   1. No error after config
//   2. Buffer state correct after concurrent different-tag input arrival
//   3. Both instructions fire (tracked via fu_busy register, not
//      combinational out_valid which self-clears in same cycle)
//   4. No error at end
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_temporal_pe_shared_buf;
  localparam int NUM_INPUTS           = 2;
  localparam int NUM_OUTPUTS          = 1;
  localparam int DATA_WIDTH           = 32;
  localparam int TAG_WIDTH            = 4;
  localparam int NUM_FU_TYPES         = 1;
  localparam int NUM_REGISTERS        = 0;
  localparam int NUM_INSTRUCTIONS     = 2;
  localparam int REG_FIFO_DEPTH       = 0;
  localparam int SHARED_OPERAND_BUFFER         = 1;
  localparam int OPERAND_BUFFER_SIZE  = 2;

  localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH;
  localparam int SAFE_PW = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1;

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

    // Configure two instructions: insn 0 -> tag=1, insn 1 -> tag=2
    // INSN_WIDTH for NR=0: 1 + TAG_WIDTH + 0 + 0 + NUM_OUTPUTS*TAG_WIDTH
    //   = 1 + 4 + 0 + 0 + 1*4 = 9
    cfg_data = '0;
    // Insn 0: valid=1, tag=1, res_tag=1
    cfg_data[dut.INSN_WIDTH - 1] = 1'b1;
    cfg_data[dut.INSN_WIDTH - 2 -: TAG_WIDTH] = TAG_WIDTH'(1);
    cfg_data[TAG_WIDTH - 1 : 0] = TAG_WIDTH'(1);
    // Insn 1: valid=1, tag=2, res_tag=2
    cfg_data[dut.INSN_WIDTH + dut.INSN_WIDTH - 1] = 1'b1;
    cfg_data[dut.INSN_WIDTH + dut.INSN_WIDTH - 2 -: TAG_WIDTH] = TAG_WIDTH'(2);
    cfg_data[dut.INSN_WIDTH + TAG_WIDTH - 1 : dut.INSN_WIDTH] = TAG_WIDTH'(2);
    @(posedge clk);

    // Check 1: no error after config
    if (error_valid !== 1'b0) begin : check_cfg
      $fatal(1, "unexpected error after config: code=%0d", error_code);
    end
    pass_count = pass_count + 1;

    // Check 2: Concurrent different-tag inputs are properly buffered.
    // Send tag=1 on port 0 and tag=2 on port 1 simultaneously. Shared buffer
    // must allocate separate entries without collision.
    in_data[0] = {TAG_WIDTH'(1), 32'h0000_000A};
    in_data[1] = {TAG_WIDTH'(2), 32'h0000_000B};
    in_valid = 2'b11;
    @(posedge clk);
    in_valid = '0;
    @(posedge clk);
    // After buffering: entry for tag=1 should have op[0] valid,
    // entry for tag=2 should have op[1] valid. No collision = no error.
    if (error_valid !== 1'b0) begin : check_no_err
      $fatal(1, "unexpected error during shared-buf concurrent input: code=%0d", error_code);
    end
    // Verify buffer entries via hierarchical access
    // Entry 0 should have tag=1 with op[0]=1,op[1]=0
    // Entry 1 should have tag=2 with op[0]=0,op[1]=1
    if (dut.g_shared_buf.sb_tag[0] != TAG_WIDTH'(1) ||
        dut.g_shared_buf.sb_op_valid[0] != 2'b01) begin : check_buf0
      $fatal(1, "shared buffer entry 0: expected tag=1 op_valid=01, got tag=%0d op_valid=%b",
             dut.g_shared_buf.sb_tag[0], dut.g_shared_buf.sb_op_valid[0]);
    end
    if (dut.g_shared_buf.sb_tag[1] != TAG_WIDTH'(2) ||
        dut.g_shared_buf.sb_op_valid[1] != 2'b10) begin : check_buf1
      $fatal(1, "shared buffer entry 1: expected tag=2 op_valid=10, got tag=%0d op_valid=%b",
             dut.g_shared_buf.sb_tag[1], dut.g_shared_buf.sb_op_valid[1]);
    end
    pass_count = pass_count + 1;

    // Check 3: Complete both instructions by sending remaining operands.
    // Send tag=2 on port 0 and tag=1 on port 1. Both buffer entries become
    // complete. Both instructions fire (back-to-back) and clear their entries.
    in_data[0] = {TAG_WIDTH'(2), 32'h0000_000C};
    in_data[1] = {TAG_WIDTH'(1), 32'h0000_000D};
    in_valid = 2'b11;
    @(posedge clk);
    in_valid = '0;

    // Wait for both fires to complete. After firing, buffer entries are
    // invalidated. Check that both entries are cleared.
    repeat (5) @(posedge clk);

    // Both entries should be invalidated (both instructions fired)
    if (|dut.g_shared_buf.sb_op_valid[0] || |dut.g_shared_buf.sb_op_valid[1]) begin : check_fired
      $fatal(1, "shared buffer not cleared after both fires: entry0=%b entry1=%b",
             dut.g_shared_buf.sb_op_valid[0], dut.g_shared_buf.sb_op_valid[1]);
    end
    pass_count = pass_count + 1;

    // Check 4: still no error
    if (error_valid !== 1'b0) begin : check_final
      $fatal(1, "unexpected error after shared-buf test: code=%0d", error_code);
    end
    pass_count = pass_count + 1;

    $display("PASS: tb_temporal_pe_shared_buf (%0d checks)", pass_count);
    $finish;
  end

  initial begin : timeout
    #20000;
    $fatal(1, "TIMEOUT");
  end
endmodule
