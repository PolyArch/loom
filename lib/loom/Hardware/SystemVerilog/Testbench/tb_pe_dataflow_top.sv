//===-- tb_pe_dataflow_top.sv - E2E test for dataflow invariant PE -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

module tb_pe_dataflow_top;

  logic        clk;
  logic        rst_n;
  logic        d_valid;
  logic        d_ready;
  logic        d_data;
  logic        a_valid;
  logic        a_ready;
  logic [31:0] a_data;
  logic        o_valid;
  logic        o_ready;
  logic [31:0] o_data;

  pe_dataflow_top dut (
    .clk    (clk),
    .rst_n  (rst_n),
    .d_valid(d_valid),
    .d_ready(d_ready),
    .d_data (d_data),
    .a_valid(a_valid),
    .a_ready(a_ready),
    .a_data (a_data),
    .o_valid(o_valid),
    .o_ready(o_ready),
    .o_data (o_data)
  );

  initial begin : clk_gen
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_pe_dataflow_top);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_pe_dataflow_top, "+mda");
  end
`endif

  task automatic load_invariant(input logic [31:0] value);
    integer iter_var0;
    logic ready_seen;
    begin : drive
      ready_seen = 1'b0;
      iter_var0 = 0;
      while (iter_var0 < 40 && !ready_seen) begin : wait_ready
        @(posedge clk);
        if (a_ready)
          ready_seen = 1'b1;
        iter_var0 = iter_var0 + 1;
      end
      if (!ready_seen) begin : timeout_ready
        $fatal(1, "a_ready did not assert before loading invariant value 0x%08h", value);
      end

      // dataflow.invariant consumes `a` when `a_valid` is sampled in S_LOAD.
      // Drive a one-cycle pulse so both simulators observe the same transition.
      @(negedge clk);
      a_data = value;
      a_valid = 1'b1;
      @(posedge clk);
      @(negedge clk);
      a_valid = 1'b0;

      // After loading, the PE should be in repeat mode where `d` drives output.
      ready_seen = 1'b0;
      iter_var0 = 0;
      while (iter_var0 < 20 && !ready_seen) begin : wait_repeat_mode
        @(posedge clk);
        if (d_ready)
          ready_seen = 1'b1;
        iter_var0 = iter_var0 + 1;
      end
      if (!ready_seen) begin : timeout_repeat_mode
        $fatal(1, "dataflow.invariant did not enter repeat mode after loading 0x%08h", value);
      end
    end
  endtask

  task automatic expect_repeat(input logic [31:0] expected);
    integer iter_var0;
    logic seen;
    begin : wait_out
      seen = 1'b0;
      iter_var0 = 0;
      while (iter_var0 < 40 && !seen) begin : loop
        @(posedge clk);
        if (o_valid) begin : got_out
          if (o_data !== expected) begin : bad_data
            $fatal(1, "expected repeated value 0x%08h, got 0x%08h", expected, o_data);
          end
          seen = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!seen) begin : timeout_out
        $fatal(1, "timeout waiting repeated value 0x%08h", expected);
      end
    end
  endtask

  initial begin : main
    integer pass_count;
    pass_count = 0;

    rst_n   = 1'b0;
    d_valid = 1'b0;
    d_data  = 1'b0;
    a_valid = 1'b0;
    a_data  = '0;
    o_ready = 1'b1;

    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    if (o_valid !== 1'b0) begin : check_reset
      $fatal(1, "o_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    load_invariant(32'h0000_002A);
    pass_count = pass_count + 1;

    // Generate repeat triggers.
    d_data = 1'b1;
    d_valid = 1'b1;
    expect_repeat(32'h0000_002A);
    pass_count = pass_count + 1;

    expect_repeat(32'h0000_002A);
    pass_count = pass_count + 1;

    // Backpressure must block trigger consumption.
    o_ready = 1'b0;
    @(posedge clk);
    #1;
    if (d_ready !== 1'b0) begin : check_backpressure
      $fatal(1, "d_ready should be 0 when output is backpressured");
    end

    o_ready = 1'b1;
    expect_repeat(32'h0000_002A);
    pass_count = pass_count + 1;

    d_valid = 1'b0;
    @(posedge clk);

    $display("PASS: tb_pe_dataflow_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : timeout
    #200000;
    $fatal(1, "TIMEOUT");
  end

endmodule
