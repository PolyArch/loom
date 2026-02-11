//===-- tb_pe_dataflow_stream_top.sv - E2E test for dataflow stream PE -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

module tb_pe_dataflow_stream_top;

  logic        clk;
  logic        rst_n;

  logic        start_valid;
  logic        start_ready;
  logic [63:0] start_data;

  logic        step_valid;
  logic        step_ready;
  logic [63:0] step_data;

  logic        bound_valid;
  logic        bound_ready;
  logic [63:0] bound_data;

  logic        idx_valid;
  logic        idx_ready;
  logic [63:0] idx_data;

  logic        cont_valid;
  logic        cont_ready;
  logic        cont_data;

  logic [4:0]  p0_cfg_data;

  pe_dataflow_stream_top dut (
    .clk        (clk),
    .rst_n      (rst_n),
    .start_valid(start_valid),
    .start_ready(start_ready),
    .start_data (start_data),
    .step_valid (step_valid),
    .step_ready (step_ready),
    .step_data  (step_data),
    .bound_valid(bound_valid),
    .bound_ready(bound_ready),
    .bound_data (bound_data),
    .idx_valid  (idx_valid),
    .idx_ready  (idx_ready),
    .idx_data   (idx_data),
    .cont_valid (cont_valid),
    .cont_ready (cont_ready),
    .cont_data  (cont_data),
    .p0_cfg_data(p0_cfg_data)
  );

  initial begin : clk_gen
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_pe_dataflow_stream_top);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_pe_dataflow_stream_top, "+mda");
  end
`endif

  task automatic drive_triplet(
      input logic [63:0] s,
      input logic [63:0] st,
      input logic [63:0] bd
  );
    integer iter_var0;
    logic accepted;
    begin : drive_task
      @(negedge clk);
      start_data  = s;
      step_data   = st;
      bound_data  = bd;
      start_valid = 1'b1;
      step_valid  = 1'b1;
      bound_valid = 1'b1;

      accepted = 1'b0;
      iter_var0 = 0;
      while (iter_var0 < 40 && !accepted) begin : loop
        @(posedge clk);
        if (start_ready && step_ready && bound_ready)
          accepted = 1'b1;
        iter_var0 = iter_var0 + 1;
      end
      if (!accepted) begin : timeout_accept
        $fatal(1, "stream input triplet was not accepted");
      end

      @(negedge clk);
      start_valid = 1'b0;
      step_valid  = 1'b0;
      bound_valid = 1'b0;
    end
  endtask

  task automatic expect_pair(input logic [63:0] expected_idx,
                             input logic expected_cont);
    integer iter_var0;
    logic seen;
    begin : expect_task
      iter_var0 = 0;
      seen = 1'b0;
      while (iter_var0 < 60 && !seen) begin : loop
        @(posedge clk);
        if (idx_valid && cont_valid) begin : got
          if (idx_data !== expected_idx) begin : bad_idx
            $fatal(1, "idx mismatch: expected %0d got %0d", expected_idx, idx_data);
          end
          if (cont_data !== expected_cont) begin : bad_cont
            $fatal(1, "cont mismatch: expected %0b got %0b", expected_cont, cont_data);
          end
          seen = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!seen) begin : timeout_expect
        $fatal(1, "stream output timeout (idx=%0d cont=%0b)", expected_idx, expected_cont);
      end
    end
  endtask

  initial begin : main
    integer pass_count;
    integer iter_var0;
    pass_count = 0;

    rst_n = 1'b0;
    start_valid = 1'b0;
    start_data  = '0;
    step_valid  = 1'b0;
    step_data   = '0;
    bound_valid = 1'b0;
    bound_data  = '0;
    idx_ready   = 1'b1;
    cont_ready  = 1'b1;
    p0_cfg_data = 5'b00001; // '<'

    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    if (idx_valid !== 1'b0 || cont_valid !== 1'b0) begin : check_reset
      $fatal(1, "stream outputs should be invalid after reset");
    end
    pass_count = pass_count + 1;

    // start=0, step=2, bound=5 with '<' => (0,T), (2,T), (4,T), (6,F)
    drive_triplet(64'd0, 64'd2, 64'd5);
    expect_pair(64'd0, 1'b1);
    expect_pair(64'd2, 1'b1);
    expect_pair(64'd4, 1'b1);
    expect_pair(64'd6, 1'b0);
    pass_count = pass_count + 1;

    // Backpressure holds current pair stable until both outputs ready.
    drive_triplet(64'd10, 64'd3, 64'd20);
    idx_ready = 1'b0;
    cont_ready = 1'b0;
    iter_var0 = 0;
    while (iter_var0 < 40 && !(idx_valid && cont_valid)) begin : wait_first
      @(posedge clk);
      iter_var0 = iter_var0 + 1;
    end
    if (!(idx_valid && cont_valid)) begin : timeout_first
      $fatal(1, "stream did not produce first output under backpressure");
    end
    #1;
    if (idx_data !== 64'd10 || cont_data !== 1'b1) begin : bad_first
      $fatal(1, "unexpected first output under backpressure idx=%0d cont=%0b", idx_data, cont_data);
    end

    repeat (2) begin : hold_check
      @(posedge clk);
      #1;
      if (idx_data !== 64'd10 || cont_data !== 1'b1) begin : bad_hold
        $fatal(1, "stream output changed while backpressured");
      end
    end

    idx_ready = 1'b1;
    cont_ready = 1'b1;
    // start=10, step=3, bound=20, '<' => (10,T), (13,T), (16,T), (19,T), (22,F)
    expect_pair(64'd10, 1'b1);
    expect_pair(64'd13, 1'b1);
    expect_pair(64'd16, 1'b1);
    expect_pair(64'd19, 1'b1);
    expect_pair(64'd22, 1'b0);
    pass_count = pass_count + 1;

    // Zero-trip test: start=5, step=1, bound=3, '<' => (5,F)
    drive_triplet(64'd5, 64'd1, 64'd3);
    expect_pair(64'd5, 1'b0);
    pass_count = pass_count + 1;

    // != test: cfg=5'b10000, start=0, step=2, bound=6 => (0,T), (2,T), (4,T), (6,F)
    // Change cfg at negedge to avoid a race with the FSM evaluating
    // will_continue at the same posedge where the zero-trip test completes.
    @(negedge clk);
    p0_cfg_data = 5'b10000;
    drive_triplet(64'd0, 64'd2, 64'd6);
    expect_pair(64'd0, 1'b1);
    expect_pair(64'd2, 1'b1);
    expect_pair(64'd4, 1'b1);
    expect_pair(64'd6, 1'b0);
    pass_count = pass_count + 1;

    $display("PASS: tb_pe_dataflow_stream_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : timeout
    #240000;
    $fatal(1, "TIMEOUT");
  end

endmodule
