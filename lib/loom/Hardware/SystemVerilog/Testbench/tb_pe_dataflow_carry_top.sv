//===-- tb_pe_dataflow_carry_top.sv - E2E test for dataflow carry PE -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

module tb_pe_dataflow_carry_top;

  logic        clk;
  logic        rst_n;

  logic        d_valid;
  logic        d_ready;
  logic        d_data;

  logic        a_valid;
  logic        a_ready;
  logic [31:0] a_data;

  logic        b_valid;
  logic        b_ready;
  logic [31:0] b_data;

  logic        o_valid;
  logic        o_ready;
  logic [31:0] o_data;

  pe_dataflow_carry_top dut (
    .clk    (clk),
    .rst_n  (rst_n),
    .d_valid(d_valid),
    .d_ready(d_ready),
    .d_data (d_data),
    .a_valid(a_valid),
    .a_ready(a_ready),
    .a_data (a_data),
    .b_valid(b_valid),
    .b_ready(b_ready),
    .b_data (b_data),
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
    $dumpvars(0, tb_pe_dataflow_carry_top);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_pe_dataflow_carry_top, "+mda");
  end
`endif

  task automatic load_init(input logic [31:0] init_value);
    integer iter_var0;
    logic accepted;
    begin : load_task
      @(negedge clk);
      a_data = init_value;
      a_valid = 1'b1;

      accepted = 1'b0;
      iter_var0 = 0;
      while (iter_var0 < 60 && !accepted) begin : wait_hs
        @(posedge clk);
        if (o_valid && a_ready) begin : got_hs
          if (o_data !== init_value) begin : bad_data
            $fatal(1, "init mismatch: expected 0x%08h got 0x%08h", init_value, o_data);
          end
          accepted = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!accepted) begin : timeout_hs
        $fatal(1, "timeout in init phase value=0x%08h", init_value);
      end

      @(negedge clk);
      a_valid = 1'b0;
    end
  endtask

  task automatic drive_carry_once(input logic [31:0] carry_value);
    integer iter_var0;
    logic accepted;
    begin : carry_task
      @(negedge clk);
      d_data = 1'b1;
      b_data = carry_value;
      d_valid = 1'b1;
      b_valid = 1'b1;

      accepted = 1'b0;
      iter_var0 = 0;
      while (iter_var0 < 60 && !accepted) begin : wait_hs
        @(posedge clk);
        if (o_valid && d_ready && b_ready) begin : got_hs
          if (o_data !== carry_value) begin : bad_data
            $fatal(1, "carry phase mismatch: expected 0x%08h got 0x%08h", carry_value, o_data);
          end
          accepted = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!accepted) begin : timeout_hs
        $fatal(1, "timeout in carry phase value=0x%08h", carry_value);
      end

      @(negedge clk);
      d_valid = 1'b0;
      b_valid = 1'b0;
    end
  endtask

  task automatic drive_carry_done;
    begin : done_task
      // Drive d_data=0, d_valid=1 for one cycle.
      // In S_BLOCK with d_data=0, d_ready is combinationally 1 and the
      // handshake fires at the next posedge. The NBA then transitions the
      // FSM to S_INIT, making d_ready go to 0 after the NBA update.
      // Therefore we verify the transition succeeded by checking a_ready
      // (which equals o_ready=1 in S_INIT) instead of d_ready.
      @(negedge clk);
      d_data = 1'b0;
      d_valid = 1'b1;
      b_valid = 1'b0;

      @(posedge clk);
      #1;
      // After NBA, state = S_INIT: a_ready = o_ready = 1, o_valid = 0
      if (!a_ready) begin : verify_init
        $fatal(1, "carry done: FSM did not return to S_INIT (a_ready not asserted)");
      end
      if (o_valid) begin : verify_no_output
        $fatal(1, "carry done: unexpected o_valid after done transition");
      end

      @(negedge clk);
      d_valid = 1'b0;
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
    b_valid = 1'b0;
    b_data  = '0;
    o_ready = 1'b1;

    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    if (o_valid !== 1'b0) begin : check_reset
      $fatal(1, "o_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    load_init(32'h0000_0011);
    pass_count = pass_count + 1;

    drive_carry_once(32'h0000_0022);
    pass_count = pass_count + 1;

    drive_carry_once(32'h0000_0033);
    pass_count = pass_count + 1;

    // Backpressure: carry input tokens must not be consumed while output is blocked.
    @(negedge clk);
    o_ready = 1'b0;
    d_data = 1'b1;
    b_data = 32'h0000_0044;
    d_valid = 1'b1;
    b_valid = 1'b1;
    #1;
    if (d_ready !== 1'b0 || b_ready !== 1'b0) begin : check_bp
      $fatal(1, "carry backpressure mismatch: d_ready=%0b b_ready=%0b", d_ready, b_ready);
    end

    o_ready = 1'b1;
    drive_carry_once(32'h0000_0044);
    pass_count = pass_count + 1;

    // d_data=0 should consume only d, return to S_INIT
    drive_carry_done();
    pass_count = pass_count + 1;

    // After done, should be back in S_INIT - load new init value
    load_init(32'h0000_00AA);
    pass_count = pass_count + 1;

    drive_carry_once(32'h0000_00BB);
    pass_count = pass_count + 1;

    // Multi-burst: init -> carry(T) -> carry(T) -> done(F) -> init -> carry(T) -> done(F)
    drive_carry_done();
    pass_count = pass_count + 1;

    load_init(32'h0000_00CC);
    pass_count = pass_count + 1;

    drive_carry_once(32'h0000_00DD);
    pass_count = pass_count + 1;

    drive_carry_done();
    pass_count = pass_count + 1;

    $display("PASS: tb_pe_dataflow_carry_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : timeout
    #200000;
    $fatal(1, "TIMEOUT");
  end

endmodule
