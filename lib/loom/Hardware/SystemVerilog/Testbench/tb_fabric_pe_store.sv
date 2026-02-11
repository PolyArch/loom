//===-- tb_fabric_pe_store.sv - Parameterized store PE test ----*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_fabric_pe_store;
  parameter int ELEM_WIDTH  = 32;
  parameter int ADDR_WIDTH  = 16;
  parameter int TAG_WIDTH   = 2;
  parameter int HW_TYPE     = 1;
  parameter int QUEUE_DEPTH = 2;

  localparam int ADDR_PW = (ADDR_WIDTH + TAG_WIDTH > 0) ? ADDR_WIDTH + TAG_WIDTH : 1;
  localparam int ELEM_PW = (ELEM_WIDTH + TAG_WIDTH > 0) ? ELEM_WIDTH + TAG_WIDTH : 1;
  localparam int CTRL_PW = (HW_TYPE == 1 && TAG_WIDTH > 0) ? TAG_WIDTH : 1;
  localparam int CFG_PW  = (HW_TYPE == 0 && TAG_WIDTH > 0) ? TAG_WIDTH : 1;

  logic               clk;
  logic               rst_n;

  logic               in0_valid;
  logic               in0_ready;
  logic [ADDR_PW-1:0] in0_data;

  logic               in1_valid;
  logic               in1_ready;
  logic [ELEM_PW-1:0] in1_data;

  logic               in2_valid;
  logic               in2_ready;
  logic [CTRL_PW-1:0] in2_data;

  logic               out0_valid;
  logic               out0_ready;
  logic [ADDR_PW-1:0] out0_data;

  logic               out1_valid;
  logic               out1_ready;
  logic [ELEM_PW-1:0] out1_data;

  logic [CFG_PW-1:0] cfg_data;

  fabric_pe_store #(
    .ELEM_WIDTH(ELEM_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH),
    .TAG_WIDTH(TAG_WIDTH),
    .HW_TYPE(HW_TYPE),
    .QUEUE_DEPTH(QUEUE_DEPTH)
  ) dut (
    .clk(clk), .rst_n(rst_n),
    .in0_valid(in0_valid), .in0_ready(in0_ready), .in0_data(in0_data),
    .in1_valid(in1_valid), .in1_ready(in1_ready), .in1_data(in1_data),
    .in2_valid(in2_valid), .in2_ready(in2_ready), .in2_data(in2_data),
    .out0_valid(out0_valid), .out0_ready(out0_ready), .out0_data(out0_data),
    .out1_valid(out1_valid), .out1_ready(out1_ready), .out1_data(out1_data),
    .cfg_data(cfg_data)
  );

  initial begin : clk_gen
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  initial begin : test
    integer pass_count;
    integer iter_var0;
    pass_count = 0;

    rst_n = 1'b0;
    in0_valid = 1'b0;
    in0_data = '0;
    in1_valid = 1'b0;
    in1_data = '0;
    in2_valid = 1'b0;
    in2_data = '0;
    out0_ready = 1'b1;
    out1_ready = 1'b1;
    cfg_data = '0;

    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    if (HW_TYPE == 0) begin : overwrite_smoke
      if (TAG_WIDTH > 0) begin : g_tagged_cfg
        cfg_data = CFG_PW'(3);
      end

      in0_data = ADDR_PW'(9);
      in1_data = ELEM_PW'(32'h1234);
      in2_data = CTRL_PW'(1);
      in0_valid = 1'b1;
      in1_valid = 1'b1;
      in2_valid = 1'b1;
      #1;
      if (out0_valid !== 1'b1 || out1_valid !== 1'b1) begin : check_valid
        $fatal(1, "overwrite: outputs should assert when all inputs are valid");
      end
      @(posedge clk);
      in0_valid = 1'b0;
      in1_valid = 1'b0;
      in2_valid = 1'b0;
      pass_count = pass_count + 1;
    end else begin : transparent_checks
      // Queue two addresses (tags 0 and 1).
      in0_data = (ADDR_PW'(0) << ADDR_WIDTH) | ADDR_PW'(10);
      in0_valid = 1'b1;
      iter_var0 = 0;
      while (iter_var0 < 10 && !in0_ready) begin : wait_addr_ready0
        @(posedge clk);
        iter_var0 = iter_var0 + 1;
      end
      if (!in0_ready) begin : check_addr_ready0
        $fatal(1, "transparent: first address was not accepted");
      end
      @(posedge clk);
      in0_valid = 1'b0;

      in0_data = (ADDR_PW'(1) << ADDR_WIDTH) | ADDR_PW'(11);
      in0_valid = 1'b1;
      iter_var0 = 0;
      while (iter_var0 < 10 && !in0_ready) begin : wait_addr_ready1
        @(posedge clk);
        iter_var0 = iter_var0 + 1;
      end
      if (!in0_ready) begin : check_addr_ready1
        $fatal(1, "transparent: second address was not accepted");
      end
      @(posedge clk);
      in0_valid = 1'b0;

      // Depth reached: third address must be backpressured.
      in0_data = (ADDR_PW'(2) << ADDR_WIDTH) | ADDR_PW'(12);
      in0_valid = 1'b1;
      #1;
      if (in0_ready !== 1'b0) begin : check_addr_backpressure
        $fatal(1, "transparent: address queue should apply backpressure at QUEUE_DEPTH");
      end
      @(posedge clk);
      in0_valid = 1'b0;

      // Queue data for tag 1, then ctrl tag 1 -> expect one matched output.
      in1_data = (ELEM_PW'(1) << ELEM_WIDTH) | ELEM_PW'(32'h2222);
      in1_valid = 1'b1;
      iter_var0 = 0;
      while (iter_var0 < 10 && !in1_ready) begin : wait_data_ready1
        @(posedge clk);
        iter_var0 = iter_var0 + 1;
      end
      if (!in1_ready) begin : check_data_ready1
        $fatal(1, "transparent: data token for tag 1 was not accepted");
      end
      @(posedge clk);
      in1_valid = 1'b0;

      in2_data = CTRL_PW'(1);
      in2_valid = 1'b1;
      #1;
      if (out0_valid !== 1'b1 || out1_valid !== 1'b1) begin : check_match1_valid
        $fatal(1, "transparent: expected matched store for tag 1");
      end
      if (out0_data !== ((ADDR_PW'(1) << ADDR_WIDTH) | ADDR_PW'(11))) begin : check_match1_addr
        $fatal(1, "transparent: wrong matched address for tag 1");
      end
      if (out1_data !== ((ELEM_PW'(1) << ELEM_WIDTH) | ELEM_PW'(32'h2222))) begin : check_match1_data
        $fatal(1, "transparent: wrong matched data for tag 1");
      end
      @(posedge clk);
      in2_valid = 1'b0;

      // Queue data for tag 0, then ctrl tag 0 -> expect second match.
      in1_data = (ELEM_PW'(0) << ELEM_WIDTH) | ELEM_PW'(32'h1111);
      in1_valid = 1'b1;
      @(posedge clk);
      in1_valid = 1'b0;

      in2_data = CTRL_PW'(0);
      in2_valid = 1'b1;
      #1;
      if (out0_valid !== 1'b1 || out1_valid !== 1'b1) begin : check_match0_valid
        $fatal(1, "transparent: expected matched store for tag 0");
      end
      if (out0_data !== ((ADDR_PW'(0) << ADDR_WIDTH) | ADDR_PW'(10))) begin : check_match0_addr
        $fatal(1, "transparent: wrong matched address for tag 0");
      end
      if (out1_data !== ((ELEM_PW'(0) << ELEM_WIDTH) | ELEM_PW'(32'h1111))) begin : check_match0_data
        $fatal(1, "transparent: wrong matched data for tag 0");
      end
      @(posedge clk);
      in2_valid = 1'b0;

      // Atomic output gating: if one output is not ready, neither valid may assert.
      in0_data = (ADDR_PW'(2) << ADDR_WIDTH) | ADDR_PW'(12);
      in0_valid = 1'b1;
      @(posedge clk);
      in0_valid = 1'b0;

      in1_data = (ELEM_PW'(2) << ELEM_WIDTH) | ELEM_PW'(32'h3333);
      in1_valid = 1'b1;
      @(posedge clk);
      in1_valid = 1'b0;

      in2_data = CTRL_PW'(2);
      in2_valid = 1'b1;
      out0_ready = 1'b1;
      out1_ready = 1'b0;
      #1;
      if ((out0_valid && out0_ready) || (out1_valid && out1_ready)) begin : check_atomic_block
        $fatal(1, "transparent: partial output handshake observed while one ready was low");
      end

      out1_ready = 1'b1;
      #1;
      if (out0_valid !== 1'b1 || out1_valid !== 1'b1) begin : check_atomic_release
        $fatal(1, "transparent: outputs should assert together when both ready are high");
      end
      @(posedge clk);
      in2_valid = 1'b0;

      // ---- Test 5a: independent queue backpressure ----
      // Reset to clear queues.
      rst_n = 1'b0;
      in0_valid = 1'b0;
      in1_valid = 1'b0;
      in2_valid = 1'b0;
      out0_ready = 1'b0;
      out1_ready = 1'b0;
      repeat (3) @(posedge clk);
      rst_n = 1'b1;
      @(posedge clk);

      // Fill addr queue with QUEUE_DEPTH entries. Block outputs so nothing drains.
      for (iter_var0 = 0; iter_var0 < QUEUE_DEPTH; iter_var0 = iter_var0 + 1) begin : fill_addr_q
        @(negedge clk);
        in0_data = (ADDR_PW'(iter_var0) << ADDR_WIDTH) | ADDR_PW'(80 + iter_var0);
        in0_valid = 1'b1;
        @(posedge clk);
      end
      @(negedge clk);
      in0_valid = 1'b0;
      #1;

      // Addr queue full: in0_ready must be 0.
      if (in0_ready !== 1'b0) begin : check_addr_full_bp
        $fatal(1, "store backpressure: in0_ready should be 0 when addr queue is full");
      end
      // Data queue still has space: in1_ready must be 1.
      if (in1_ready !== 1'b1) begin : check_data_space
        $fatal(1, "store backpressure: in1_ready should be 1 when data queue has space");
      end
      // Ctrl queue still has space: in2_ready must be 1.
      if (in2_ready !== 1'b1) begin : check_ctrl_space
        $fatal(1, "store backpressure: in2_ready should be 1 when ctrl queue has space");
      end
      pass_count = pass_count + 1;

      // ---- Test 5b: lowest-index-first match selection ----
      // Reset to clear queues.
      rst_n = 1'b0;
      in0_valid = 1'b0;
      in1_valid = 1'b0;
      in2_valid = 1'b0;
      out0_ready = 1'b1;
      out1_ready = 1'b1;
      repeat (3) @(posedge clk);
      rst_n = 1'b1;
      @(posedge clk);

      // Block outputs so matches don't fire immediately.
      out0_ready = 1'b0;
      out1_ready = 1'b0;

      // Enqueue addr tag=1 at slot 0.
      @(negedge clk);
      in0_data = (ADDR_PW'(1) << ADDR_WIDTH) | ADDR_PW'(90);
      in0_valid = 1'b1;
      @(posedge clk);
      @(negedge clk);
      in0_valid = 1'b0;

      // Enqueue addr tag=0 at slot 1.
      @(negedge clk);
      in0_data = (ADDR_PW'(0) << ADDR_WIDTH) | ADDR_PW'(91);
      in0_valid = 1'b1;
      @(posedge clk);
      @(negedge clk);
      in0_valid = 1'b0;

      // Enqueue data tag=1.
      @(negedge clk);
      in1_data = (ELEM_PW'(1) << ELEM_WIDTH) | ELEM_PW'(32'hDD01);
      in1_valid = 1'b1;
      @(posedge clk);
      @(negedge clk);
      in1_valid = 1'b0;

      // Enqueue data tag=0.
      @(negedge clk);
      in1_data = (ELEM_PW'(0) << ELEM_WIDTH) | ELEM_PW'(32'hDD00);
      in1_valid = 1'b1;
      @(posedge clk);
      @(negedge clk);
      in1_valid = 1'b0;

      // Enqueue ctrl tag=0.
      @(negedge clk);
      in2_data = CTRL_PW'(0);
      in2_valid = 1'b1;
      @(posedge clk);
      @(negedge clk);
      in2_valid = 1'b0;

      // Enqueue ctrl tag=1.
      @(negedge clk);
      in2_data = CTRL_PW'(1);
      in2_valid = 1'b1;
      @(posedge clk);
      @(negedge clk);
      in2_valid = 1'b0;

      // All six entries now in queues. Release outputs at negedge.
      // Match_find scans addr from index 0:
      // slot 0 has tag=1, finds data tag=1 and ctrl tag=1 -> fires tag=1 first.
      @(negedge clk);
      out0_ready = 1'b1;
      out1_ready = 1'b1;
      #1;
      if (out0_valid !== 1'b1 || out1_valid !== 1'b1) begin : check_lif_valid
        $fatal(1, "lowest-index-first store: expected outputs to fire");
      end
      // out0 = addr tag=1, addr=90.
      if (out0_data !== ((ADDR_PW'(1) << ADDR_WIDTH) | ADDR_PW'(90))) begin : check_lif_addr
        $fatal(1, "lowest-index-first store: expected tag=1 addr=90, got 0x%0h", out0_data);
      end
      // out1 = data tag=1, data=0xDD01.
      if (out1_data !== ((ELEM_PW'(1) << ELEM_WIDTH) | ELEM_PW'(32'hDD01))) begin : check_lif_data
        $fatal(1, "lowest-index-first store: expected tag=1 data=0xDD01, got 0x%0h", out1_data);
      end
      @(posedge clk);
      pass_count = pass_count + 1;

      pass_count = pass_count + 1;
    end

    $display("PASS: tb_fabric_pe_store HW_TYPE=%0d TAG_WIDTH=%0d QUEUE_DEPTH=%0d (%0d checks)",
             HW_TYPE, TAG_WIDTH, QUEUE_DEPTH, pass_count);
    $finish;
  end

  initial begin : timeout
    #20000;
    $fatal(1, "TIMEOUT");
  end

endmodule
