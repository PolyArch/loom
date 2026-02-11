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

      pass_count = pass_count + 1;
    end

    $display("PASS: tb_fabric_pe_store HW_TYPE=%0d TAG_WIDTH=%0d QUEUE_DEPTH=%0d (%0d checks)",
             HW_TYPE, TAG_WIDTH, QUEUE_DEPTH, pass_count);
    $finish;
  end

  initial begin : timeout
    #5000;
    $fatal(1, "TIMEOUT");
  end

endmodule
