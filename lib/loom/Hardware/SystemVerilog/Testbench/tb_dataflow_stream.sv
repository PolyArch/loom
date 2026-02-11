//===-- tb_dataflow_stream.sv - Direct dataflow_stream validation -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_dataflow_stream;

  localparam int WIDTH = 64;

  logic             clk;
  logic             rst_n;

  logic             start_valid;
  logic             start_ready;
  logic [WIDTH-1:0] start_data;

  logic             step_valid;
  logic             step_ready;
  logic [WIDTH-1:0] step_data;

  logic             bound_valid;
  logic             bound_ready;
  logic [WIDTH-1:0] bound_data;

  logic             index_valid;
  logic             index_ready;
  logic [WIDTH-1:0] index_data;

  logic             cont_valid;
  logic             cont_ready;
  logic             cont_data;

  logic [4:0]       cfg_cont_cond_sel;

  logic             error_valid;
  logic [15:0]      error_code;

  dataflow_stream #(
    .WIDTH(WIDTH)
  ) dut (
    .clk              (clk),
    .rst_n            (rst_n),
    .start_valid      (start_valid),
    .start_ready      (start_ready),
    .start_data       (start_data),
    .step_valid       (step_valid),
    .step_ready       (step_ready),
    .step_data        (step_data),
    .bound_valid      (bound_valid),
    .bound_ready      (bound_ready),
    .bound_data       (bound_data),
    .index_valid      (index_valid),
    .index_ready      (index_ready),
    .index_data       (index_data),
    .cont_valid       (cont_valid),
    .cont_ready       (cont_ready),
    .cont_data        (cont_data),
    .cfg_cont_cond_sel(cfg_cont_cond_sel),
    .error_valid      (error_valid),
    .error_code       (error_code)
  );

  initial begin : clk_gen
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_dataflow_stream);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_dataflow_stream, "+mda");
  end
`endif

  task automatic reset_signals;
    begin : reset_task
      start_valid = 1'b0;
      start_data = '0;
      step_valid = 1'b0;
      step_data = '0;
      bound_valid = 1'b0;
      bound_data = '0;
      index_ready = 1'b1;
      cont_ready = 1'b1;
      cfg_cont_cond_sel = 5'b00001;
    end
  endtask

  task automatic drive_triplet(
      input logic [WIDTH-1:0] s,
      input logic [WIDTH-1:0] st,
      input logic [WIDTH-1:0] bd
  );
    integer iter_var0;
    logic accepted;
    begin : drive_task
      @(negedge clk);
      start_data = s;
      step_data = st;
      bound_data = bd;
      start_valid = 1'b1;
      step_valid = 1'b1;
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
        $fatal(1, "dataflow_stream input triplet was not accepted");
      end

      @(negedge clk);
      start_valid = 1'b0;
      step_valid = 1'b0;
      bound_valid = 1'b0;
    end
  endtask

  task automatic expect_pair(
      input logic [WIDTH-1:0] expected_idx,
      input logic expected_cont
  );
    integer iter_var0;
    logic seen;
    begin : expect_task
      iter_var0 = 0;
      seen = 1'b0;
      while (iter_var0 < 60 && !seen) begin : loop
        @(posedge clk);
        if (index_valid && cont_valid) begin : got
          if (index_data !== expected_idx) begin : bad_idx
            $fatal(1, "index mismatch: expected %0d got %0d", expected_idx, index_data);
          end
          if (cont_data !== expected_cont) begin : bad_cont
            $fatal(1, "cont mismatch: expected %0b got %0b", expected_cont, cont_data);
          end
          seen = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!seen) begin : timeout_expect
        $fatal(1, "timeout waiting stream pair idx=%0d cont=%0b", expected_idx, expected_cont);
      end
    end
  endtask

  task automatic wait_error(input logic [15:0] expected_code);
    integer iter_var0;
    logic seen;
    begin : err_task
      iter_var0 = 0;
      seen = 1'b0;
      while (iter_var0 < 40 && !seen) begin : loop
        @(posedge clk);
        if (error_valid) begin : got
          if (error_code !== expected_code) begin : bad_code
            $fatal(1, "error code mismatch: expected %0d got %0d", expected_code, error_code);
          end
          seen = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!seen) begin : timeout_err
        $fatal(1, "timeout waiting error code %0d", expected_code);
      end
    end
  endtask

  initial begin : main
    integer pass_count;
    pass_count = 0;

    rst_n = 1'b0;
    reset_signals();

    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    if (error_valid !== 1'b0) begin : check_reset
      $fatal(1, "error_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    // Functional sequence: start=0 step=2 bound=5 with '<'
    // => (0,T), (2,T), (4,T), (6,F)
    drive_triplet(64'd0, 64'd2, 64'd5);
    expect_pair(64'd0, 1'b1);
    expect_pair(64'd2, 1'b1);
    expect_pair(64'd4, 1'b1);
    expect_pair(64'd6, 1'b0);
    pass_count = pass_count + 1;

    // Runtime error: zero step during running state.
    rst_n = 1'b0;
    reset_signals();
    repeat (2) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    drive_triplet(64'd1, 64'd0, 64'd8);
    wait_error(RT_DATAFLOW_STREAM_ZERO_STEP);
    pass_count = pass_count + 1;

    // Config error: non-onehot selector.
    rst_n = 1'b0;
    reset_signals();
    cfg_cont_cond_sel = 5'b00000;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;
    wait_error(CFG_PE_STREAM_CONT_COND_ONEHOT);
    pass_count = pass_count + 1;

    $display("PASS: tb_dataflow_stream (%0d checks)", pass_count);
    $finish;
  end

  initial begin : timeout
    #200000;
    $fatal(1, "TIMEOUT");
  end

endmodule
