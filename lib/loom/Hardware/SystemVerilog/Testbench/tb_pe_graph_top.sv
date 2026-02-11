//===-- tb_pe_graph_top.sv - E2E test for generated pe_graph_top -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

module tb_pe_graph_top;

  logic        clk;
  logic        rst_n;

  logic        a_valid;
  logic        a_ready;
  logic [31:0] a_data;
  logic        b_valid;
  logic        b_ready;
  logic [31:0] b_data;
  logic        c_valid;
  logic        c_ready;
  logic [31:0] c_data;
  logic        d_valid;
  logic        d_ready;
  logic [31:0] d_data;

  logic        out0_valid;
  logic        out0_ready;
  logic [31:0] out0_data;
  logic        out1_valid;
  logic        out1_ready;
  logic [31:0] out1_data;

  logic [3:0]  sw0_cfg_route_table;
  logic        error_valid;
  logic [15:0] error_code;

  pe_graph_top dut (
    .clk                (clk),
    .rst_n              (rst_n),
    .a_valid            (a_valid),
    .a_ready            (a_ready),
    .a_data             (a_data),
    .b_valid            (b_valid),
    .b_ready            (b_ready),
    .b_data             (b_data),
    .c_valid            (c_valid),
    .c_ready            (c_ready),
    .c_data             (c_data),
    .d_valid            (d_valid),
    .d_ready            (d_ready),
    .d_data             (d_data),
    .out0_valid         (out0_valid),
    .out0_ready         (out0_ready),
    .out0_data          (out0_data),
    .out1_valid         (out1_valid),
    .out1_ready         (out1_ready),
    .out1_data          (out1_data),
    .sw0_cfg_route_table(sw0_cfg_route_table),
    .error_valid        (error_valid),
    .error_code         (error_code)
  );

  initial begin : clk_gen
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_pe_graph_top);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_pe_graph_top, "+mda");
  end
`endif

  task automatic drive_inputs(
      input logic [31:0] a_v,
      input logic [31:0] b_v,
      input logic [31:0] c_v,
      input logic [31:0] d_v
  );
    integer iter_var0;
    logic accepted;
    begin : drive
      a_data  = a_v;
      b_data  = b_v;
      c_data  = c_v;
      d_data  = d_v;
      a_valid = 1'b1;
      b_valid = 1'b1;
      c_valid = 1'b1;
      d_valid = 1'b1;
      iter_var0 = 0;
      accepted = 1'b0;
      while (iter_var0 < 80 && !accepted) begin : wait_accept
        @(posedge clk);
        if (a_ready && b_ready && c_ready && d_ready) begin : got_accept
          a_valid = 1'b0;
          b_valid = 1'b0;
          c_valid = 1'b0;
          d_valid = 1'b0;
          accepted = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!accepted) begin : timeout_accept
        $fatal(1, "input acceptance timeout");
      end
    end
  endtask

  task automatic drive_alt_inputs(
      input logic [31:0] c_v,
      input logic [31:0] d_v
  );
    integer iter_var0;
    logic accepted;
    begin : drive
      c_data  = c_v;
      d_data  = d_v;
      a_data  = '0;
      b_data  = '0;
      a_valid = 1'b0;
      b_valid = 1'b0;
      c_valid = 1'b1;
      d_valid = 1'b1;
      iter_var0 = 0;
      accepted = 1'b0;
      while (iter_var0 < 80 && !accepted) begin : wait_accept
        @(posedge clk);
        if (c_ready && d_ready) begin : got_accept
          c_valid = 1'b0;
          d_valid = 1'b0;
          accepted = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!accepted) begin : timeout_accept
        $fatal(1, "alt-input acceptance timeout");
      end
    end
  endtask

  task automatic expect_out1(input logic [31:0] expected);
    integer iter_var0;
    logic seen;
    begin : expect_out1_task
      iter_var0 = 0;
      seen = 1'b0;
      if (out1_valid) begin : got_out_now
        if (out1_data !== expected) begin : bad_out_now
          $fatal(1, "out1 expected 0x%08h, got 0x%08h", expected, out1_data);
        end
        seen = 1'b1;
      end
      while (iter_var0 < 120 && !seen) begin : wait_out
        @(posedge clk);
        if (out1_valid) begin : got_out
          if (out1_data !== expected) begin : bad_out
            $fatal(1, "out1 expected 0x%08h, got 0x%08h", expected, out1_data);
          end
          seen = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!seen) begin : timeout_out
        $fatal(1, "out1 timeout (expected 0x%08h)", expected);
      end
    end
  endtask

  task automatic expect_both(
      input logic [31:0] expected0,
      input logic [31:0] expected1
  );
    integer iter_var0;
    logic seen0;
    logic seen1;
    begin : expect_both_task
      iter_var0 = 0;
      seen0 = 1'b0;
      seen1 = 1'b0;
      if (out0_valid) begin : got_out0_now
        if (out0_data !== expected0) begin : bad_out0_now
          $fatal(1, "out0 expected 0x%08h, got 0x%08h", expected0, out0_data);
        end
        seen0 = 1'b1;
      end
      if (out1_valid) begin : got_out1_now
        if (out1_data !== expected1) begin : bad_out1_now
          $fatal(1, "out1 expected 0x%08h, got 0x%08h", expected1, out1_data);
        end
        seen1 = 1'b1;
      end
      while (iter_var0 < 140 && !(seen0 && seen1)) begin : wait_pair
        @(posedge clk);
        if (out0_valid && !seen0) begin : got_out0
          if (out0_data !== expected0) begin : bad_out0
            $fatal(1, "out0 expected 0x%08h, got 0x%08h", expected0, out0_data);
          end
          seen0 = 1'b1;
        end
        if (out1_valid && !seen1) begin : got_out1
          if (out1_data !== expected1) begin : bad_out1
            $fatal(1, "out1 expected 0x%08h, got 0x%08h", expected1, out1_data);
          end
          seen1 = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!(seen0 && seen1)) begin : timeout_pair
        $fatal(1, "pair timeout out0=0x%08h out1=0x%08h seen0=%0b seen1=%0b",
               expected0, expected1, seen0, seen1);
      end
    end
  endtask

  initial begin : main
    integer pass_count;
    integer iter_var0;
    pass_count = 0;

    rst_n = 1'b0;
    a_valid = 1'b0;
    b_valid = 1'b0;
    c_valid = 1'b0;
    d_valid = 1'b0;
    a_data = '0;
    b_data = '0;
    c_data = '0;
    d_data = '0;

    out0_ready = 1'b1;
    out1_ready = 1'b1;
    sw0_cfg_route_table = 4'b1001; // out0<-in0(mul), out1<-in1(add)

    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    if (error_valid !== 1'b0) begin : check_reset
      $fatal(1, "error_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    // Test 1: diagonal mapping
    // mul=3*4=12, add=5+6=11
    drive_inputs(32'd3, 32'd4, 32'd5, 32'd6);
    expect_both(32'd12, 32'd11);
    pass_count = pass_count + 1;

    // Test 2: swap mapping
    sw0_cfg_route_table = 4'b0110; // out0<-in1(add), out1<-in0(mul)
    @(posedge clk);

    // mul=2*8=16, add=3+7=10
    drive_inputs(32'd2, 32'd8, 32'd3, 32'd7);
    expect_both(32'd10, 32'd16);
    pass_count = pass_count + 1;

    // Test 3: add-branch only under out0 backpressure
    rst_n = 1'b0;
    a_valid = 1'b0;
    b_valid = 1'b0;
    c_valid = 1'b0;
    d_valid = 1'b0;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    sw0_cfg_route_table = 4'b1001;
    out0_ready = 1'b0;
    out1_ready = 1'b1;
    @(posedge clk);

    // add=4+5=9
    drive_alt_inputs(32'd4, 32'd5);
    expect_out1(32'd9);

    iter_var0 = 0;
    while (iter_var0 < 8) begin : check_out0_quiet
      @(posedge clk);
      if (out0_valid !== 1'b0) begin : bad_out0
        $fatal(1, "out0 should stay idle while only add branch is driven");
      end
      iter_var0 = iter_var0 + 1;
    end

    out0_ready = 1'b1;
    pass_count = pass_count + 1;

    if (error_valid !== 1'b0) begin : check_no_error
      $fatal(1, "unexpected error: code=%0d", error_code);
    end

    $display("PASS: tb_pe_graph_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : timeout
    #300000;
    $fatal(1, "TIMEOUT");
  end

endmodule
