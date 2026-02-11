//===-- tb_pe_fork_top.sv - E2E test for exported pe_fork_top ---*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

module tb_pe_fork_top;

  logic        clk;
  logic        rst_n;
  logic        in0_valid;
  logic        in0_ready;
  logic [31:0] in0_data;
  logic        in1_valid;
  logic        in1_ready;
  logic [31:0] in1_data;
  logic        in2_valid;
  logic        in2_ready;
  logic [31:0] in2_data;
  logic        out0_valid;
  logic        out0_ready;
  logic [31:0] out0_data;
  logic        out1_valid;
  logic        out1_ready;
  logic [31:0] out1_data;

  pe_fork_top dut (
    .clk       (clk),
    .rst_n     (rst_n),
    .in0_valid (in0_valid),
    .in0_ready (in0_ready),
    .in0_data  (in0_data),
    .in1_valid (in1_valid),
    .in1_ready (in1_ready),
    .in1_data  (in1_data),
    .in2_valid (in2_valid),
    .in2_ready (in2_ready),
    .in2_data  (in2_data),
    .out0_valid(out0_valid),
    .out0_ready(out0_ready),
    .out0_data (out0_data),
    .out1_valid(out1_valid),
    .out1_ready(out1_ready),
    .out1_data (out1_data)
  );

  initial begin : clk_gen
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_pe_fork_top);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_pe_fork_top, "+mda");
  end
`endif

  task automatic drive_inputs(
      input logic [31:0] a,
      input logic [31:0] b,
      input logic [31:0] c
  );
    integer iter_var0;
    logic accepted;
    begin : drive
      in0_data  = a;
      in1_data  = b;
      in2_data  = c;
      in0_valid = 1'b1;
      in1_valid = 1'b1;
      in2_valid = 1'b1;
      iter_var0 = 0;
      accepted  = 1'b0;
      while (iter_var0 < 40 && !accepted) begin : wait_accept
        @(posedge clk);
        if (in0_ready && in1_ready && in2_ready) begin : got_accept
          in0_valid = 1'b0;
          in1_valid = 1'b0;
          in2_valid = 1'b0;
          accepted  = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!accepted) begin : timeout_accept
        $fatal(1, "input handshake timeout (in0=%0b in1=%0b in2=%0b)",
               in0_ready, in1_ready, in2_ready);
      end
    end
  endtask

  task automatic expect_outputs(
      input logic [31:0] expected0,
      input logic [31:0] expected1
  );
    integer iter_var0;
    logic seen0;
    logic seen1;
    begin : expect_outputs_task
      iter_var0 = 0;
      seen0 = 1'b0;
      seen1 = 1'b0;
      if (out0_valid) begin : got_output0_now
        if (out0_data !== expected0) begin : mismatch0_now
          $fatal(1, "out0 expected 0x%08h, got 0x%08h", expected0, out0_data);
        end
        seen0 = 1'b1;
      end
      if (out1_valid) begin : got_output1_now
        if (out1_data !== expected1) begin : mismatch1_now
          $fatal(1, "out1 expected 0x%08h, got 0x%08h", expected1, out1_data);
        end
        seen1 = 1'b1;
      end
      while (iter_var0 < 80 && !(seen0 && seen1)) begin : wait_out
        @(posedge clk);
        if (out0_valid && !seen0) begin : got_output0
          if (out0_data !== expected0) begin : mismatch0
            $fatal(1, "out0 expected 0x%08h, got 0x%08h", expected0, out0_data);
          end
          seen0 = 1'b1;
        end
        if (out1_valid && !seen1) begin : got_output1
          if (out1_data !== expected1) begin : mismatch1
            $fatal(1, "out1 expected 0x%08h, got 0x%08h", expected1, out1_data);
          end
          seen1 = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!(seen0 && seen1)) begin : timeout_out
        $fatal(1, "output timeout (out0=0x%08h out1=0x%08h, seen0=%0b seen1=%0b)",
               expected0, expected1, seen0, seen1);
      end
    end
  endtask

  initial begin : main
    integer pass_count;
    pass_count = 0;

    rst_n     = 1'b0;
    in0_valid = 1'b0;
    in1_valid = 1'b0;
    in2_valid = 1'b0;
    in0_data  = '0;
    in1_data  = '0;
    in2_data  = '0;
    out0_ready = 1'b1;
    out1_ready = 1'b1;

    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    if (out0_valid !== 1'b0 || out1_valid !== 1'b0) begin : check_reset
      $fatal(1, "out0_valid/out1_valid should be 0 right after reset");
    end
    pass_count = pass_count + 1;

    // f0(a, b) = a + b, f1(a, c) = a - c
    drive_inputs(32'd10, 32'd3, 32'd4);
    expect_outputs(32'd13, 32'd6);
    pass_count = pass_count + 1;

    drive_inputs(32'd7, 32'd9, 32'd2);
    expect_outputs(32'd16, 32'd5);
    pass_count = pass_count + 1;

    drive_inputs(32'hFFFF_FFFF, 32'd1, 32'd2);
    expect_outputs(32'd0, 32'hFFFF_FFFD);
    pass_count = pass_count + 1;

    $display("PASS: tb_pe_fork_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : timeout
    #200000;
    $fatal(1, "TIMEOUT");
  end

endmodule
