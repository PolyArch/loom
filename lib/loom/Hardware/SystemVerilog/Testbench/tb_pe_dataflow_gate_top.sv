//===-- tb_pe_dataflow_gate_top.sv - E2E test for dataflow gate PE -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

module tb_pe_dataflow_gate_top;

  logic        clk;
  logic        rst_n;

  logic        bv_valid;
  logic        bv_ready;
  logic [31:0] bv_data;

  logic        bc_valid;
  logic        bc_ready;
  logic        bc_data;

  logic        av_valid;
  logic        av_ready;
  logic [31:0] av_data;

  logic        ac_valid;
  logic        ac_ready;
  logic        ac_data;

  pe_dataflow_gate_top dut (
    .clk    (clk),
    .rst_n  (rst_n),
    .bv_valid(bv_valid),
    .bv_ready(bv_ready),
    .bv_data (bv_data),
    .bc_valid(bc_valid),
    .bc_ready(bc_ready),
    .bc_data (bc_data),
    .av_valid(av_valid),
    .av_ready(av_ready),
    .av_data (av_data),
    .ac_valid(ac_valid),
    .ac_ready(ac_ready),
    .ac_data (ac_data)
  );

  initial begin : clk_gen
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_pe_dataflow_gate_top);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_pe_dataflow_gate_top, "+mda");
  end
`endif

  task automatic drive_pair_expect(input logic [31:0] v, input logic c);
    integer iter_var0;
    logic seen;
    begin : pair_task
      @(negedge clk);
      bv_data  = v;
      bc_data  = c;
      bv_valid = 1'b1;
      bc_valid = 1'b1;

      seen = 1'b0;
      iter_var0 = 0;
      while (iter_var0 < 40 && !seen) begin : loop
        #1;
        if (av_valid && ac_valid && bv_ready && bc_ready) begin : got
          if (av_data !== v) begin : bad_val
            $fatal(1, "gate av mismatch: expected 0x%08h got 0x%08h", v, av_data);
          end
          if (ac_data !== c) begin : bad_cond
            $fatal(1, "gate ac mismatch: expected %0b got %0b", c, ac_data);
          end
          seen = 1'b1;
        end
        @(posedge clk);
        iter_var0 = iter_var0 + 1;
      end
      if (!seen) begin : timeout_pair
        $fatal(1, "gate pair output timeout");
      end

      @(negedge clk);
      bv_valid = 1'b0;
      bc_valid = 1'b0;
    end
  endtask

  initial begin : main
    integer pass_count;
    integer iter_var0;
    logic consumed_head;
    pass_count = 0;

    rst_n = 1'b0;
    bv_valid = 1'b0;
    bv_data  = '0;
    bc_valid = 1'b0;
    bc_data  = 1'b0;
    av_ready = 1'b1;
    ac_ready = 1'b1;

    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    if (av_valid !== 1'b0 || ac_valid !== 1'b0) begin : check_reset
      $fatal(1, "gate outputs should be invalid right after reset");
    end
    #1;
    if (bc_ready !== 1'b1 || bv_ready !== 1'b0) begin : check_skip_state
      $fatal(1, "gate skip-head ready mismatch: bc_ready=%0b bv_ready=%0b", bc_ready, bv_ready);
    end
    pass_count = pass_count + 1;

    // First condition token is consumed and dropped in skip-head state.
    bc_data = 1'b1;
    bc_valid = 1'b1;
    consumed_head = 1'b0;
    iter_var0 = 0;
    while (iter_var0 < 20 && !consumed_head) begin : wait_head
      @(posedge clk);
      if (bc_ready)
        consumed_head = 1'b1;
      iter_var0 = iter_var0 + 1;
    end
    if (!consumed_head) begin : timeout_head
      $fatal(1, "gate skip-head token was not consumed");
    end
    @(negedge clk);
    bc_valid = 1'b0;

    drive_pair_expect(32'h0000_00AA, 1'b1);
    pass_count = pass_count + 1;

    // Backpressure in one output blocks both inputs.
    @(negedge clk);
    ac_ready = 1'b0;
    bv_data  = 32'h0000_00BB;
    bc_data  = 1'b0;
    bv_valid = 1'b1;
    bc_valid = 1'b1;
    #1;
    if (bv_ready !== 1'b0 || bc_ready !== 1'b0) begin : check_backpressure
      $fatal(1, "gate backpressure mismatch: bv_ready=%0b bc_ready=%0b", bv_ready, bc_ready);
    end

    ac_ready = 1'b1;
    #1;
    if (av_valid !== 1'b1 || ac_valid !== 1'b1) begin : check_release
      $fatal(1, "gate outputs should assert after releasing backpressure");
    end
    @(posedge clk);
    @(negedge clk);
    bv_valid = 1'b0;
    bc_valid = 1'b0;
    pass_count = pass_count + 1;

    $display("PASS: tb_pe_dataflow_gate_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : timeout
    #200000;
    $fatal(1, "TIMEOUT");
  end

endmodule
