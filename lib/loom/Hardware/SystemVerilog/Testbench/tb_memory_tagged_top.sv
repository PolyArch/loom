//===-- tb_memory_tagged_top.sv - E2E tagged memory/load/store test -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

module tb_memory_tagged_top;

  logic        clk;
  logic        rst_n;

  logic        ld_addr_valid;
  logic        ld_addr_ready;
  logic [64:0] ld_addr_data;
  logic        ld_ctrl_valid;
  logic        ld_ctrl_ready;
  logic        ld_ctrl_data;

  logic        st_addr_valid;
  logic        st_addr_ready;
  logic [64:0] st_addr_data;
  logic        st_data_valid;
  logic        st_data_ready;
  logic [32:0] st_data_data;
  logic        st_ctrl_valid;
  logic        st_ctrl_ready;
  logic        st_ctrl_data;

  logic        ld_out_valid;
  logic        ld_out_ready;
  logic [32:0] ld_out_data;

  logic        lddone_valid;
  logic        lddone_ready;
  logic        lddone_data;
  logic        stdone_valid;
  logic        stdone_ready;
  logic        stdone_data;

  // Config port: TAG_WIDTH=1, ADDR_WIDTH=64, NUM_REGION=1
  // REGION_ENTRY_WIDTH = 1 + 1 + (1+1) + 64 = 68
  // Layout: [valid(1)] [start_tag(1)] [end_tag(2)] [addr_offset(64)]
  logic [67:0] m0_cfg_data;

  logic        error_valid;
  logic [15:0] error_code;

  memory_tagged_top dut (
    .clk         (clk),
    .rst_n       (rst_n),
    .ld_addr_valid(ld_addr_valid),
    .ld_addr_ready(ld_addr_ready),
    .ld_addr_data (ld_addr_data),
    .ld_ctrl_valid(ld_ctrl_valid),
    .ld_ctrl_ready(ld_ctrl_ready),
    .ld_ctrl_data (ld_ctrl_data),
    .st_addr_valid(st_addr_valid),
    .st_addr_ready(st_addr_ready),
    .st_addr_data (st_addr_data),
    .st_data_valid(st_data_valid),
    .st_data_ready(st_data_ready),
    .st_data_data (st_data_data),
    .st_ctrl_valid(st_ctrl_valid),
    .st_ctrl_ready(st_ctrl_ready),
    .st_ctrl_data (st_ctrl_data),
    .ld_out_valid (ld_out_valid),
    .ld_out_ready (ld_out_ready),
    .ld_out_data  (ld_out_data),
    .lddone_valid (lddone_valid),
    .lddone_ready (lddone_ready),
    .lddone_data  (lddone_data),
    .stdone_valid (stdone_valid),
    .stdone_ready (stdone_ready),
    .stdone_data  (stdone_data),
    .m0_cfg_data  (m0_cfg_data),
    .error_valid  (error_valid),
    .error_code   (error_code)
  );

  initial begin : clk_gen
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_memory_tagged_top);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_memory_tagged_top, "+mda");
  end
`endif

  function automatic logic [64:0] pack_addr(
      input logic tag,
      input logic [63:0] addr
  );
    pack_addr = {tag, addr};
  endfunction

  function automatic logic [32:0] pack_data(
      input logic tag,
      input logic [31:0] data
  );
    pack_data = {tag, data};
  endfunction

  task automatic drive_store(
      input logic tag,
      input logic [63:0] addr,
      input logic [31:0] data
  );
    integer iter_var0;
    logic accepted;
    begin : drive
      st_addr_data  = pack_addr(tag, addr);
      st_data_data  = pack_data(tag, data);
      st_ctrl_data  = tag;
      st_addr_valid = 1'b1;
      st_data_valid = 1'b1;
      st_ctrl_valid = 1'b1;

      iter_var0 = 0;
      accepted = 1'b0;
      while (iter_var0 < 80 && !accepted) begin : wait_accept
        @(posedge clk);
        if (st_addr_ready && st_data_ready && st_ctrl_ready) begin : got_accept
          st_addr_valid = 1'b0;
          st_data_valid = 1'b0;
          st_ctrl_valid = 1'b0;
          accepted = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!accepted) begin : timeout_accept
        $fatal(1, "store handshake timeout for tag=%0d addr=%0d", tag, addr);
      end
    end
  endtask

  task automatic wait_stdone(input logic expected_tag);
    integer iter_var0;
    logic seen;
    begin : wait_done
      iter_var0 = 0;
      seen = 1'b0;
      if (stdone_valid) begin : got_done_now
        if (stdone_data !== expected_tag) begin : bad_tag_now
          $fatal(1, "stdone tag mismatch: expected %0d got %0d",
                 expected_tag, stdone_data);
        end
        seen = 1'b1;
      end
      while (iter_var0 < 80 && !seen) begin : loop
        @(posedge clk);
        if (stdone_valid) begin : got_done
          if (stdone_data !== expected_tag) begin : bad_tag
            $fatal(1, "stdone tag mismatch: expected %0d got %0d",
                   expected_tag, stdone_data);
          end
          seen = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!seen) begin : timeout_done
        $fatal(1, "stdone timeout for tag %0d", expected_tag);
      end
    end
  endtask

  task automatic drive_ld_addr(input logic tag, input logic [63:0] addr);
    integer iter_var0;
    logic accepted;
    begin : drive
      ld_addr_data  = pack_addr(tag, addr);
      ld_addr_valid = 1'b1;
      iter_var0 = 0;
      accepted = 1'b0;
      while (iter_var0 < 80 && !accepted) begin : wait_accept
        @(posedge clk);
        if (ld_addr_ready) begin : got_accept
          ld_addr_valid = 1'b0;
          accepted = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!accepted) begin : timeout_accept
        $fatal(1, "ld_addr handshake timeout for tag=%0d addr=%0d", tag, addr);
      end
    end
  endtask

  task automatic drive_ld_ctrl(input logic tag);
    integer iter_var0;
    logic accepted;
    begin : drive
      ld_ctrl_data  = tag;
      ld_ctrl_valid = 1'b1;
      iter_var0 = 0;
      accepted = 1'b0;
      while (iter_var0 < 80 && !accepted) begin : wait_accept
        @(posedge clk);
        if (ld_ctrl_ready) begin : got_accept
          ld_ctrl_valid = 1'b0;
          accepted = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!accepted) begin : timeout_accept
        $fatal(1, "ld_ctrl handshake timeout for tag=%0d", tag);
      end
    end
  endtask

  task automatic wait_ld_out(
      input logic expected_tag,
      input logic [31:0] expected_data
  );
    integer iter_var0;
    logic seen;
    begin : wait_out
      iter_var0 = 0;
      seen = 1'b0;
      if (ld_out_valid) begin : got_out_now
        if (ld_out_data[32] !== expected_tag) begin : bad_tag_now
          $fatal(1, "ld_out tag mismatch: expected %0d got %0d",
                 expected_tag, ld_out_data[32]);
        end
        if (ld_out_data[31:0] !== expected_data) begin : bad_data_now
          $fatal(1, "ld_out data mismatch: expected 0x%08h got 0x%08h",
                 expected_data, ld_out_data[31:0]);
        end
        seen = 1'b1;
      end
      while (iter_var0 < 120 && !seen) begin : loop
        @(posedge clk);
        if (ld_out_valid) begin : got_out
          if (ld_out_data[32] !== expected_tag) begin : bad_tag
            $fatal(1, "ld_out tag mismatch: expected %0d got %0d",
                   expected_tag, ld_out_data[32]);
          end
          if (ld_out_data[31:0] !== expected_data) begin : bad_data
            $fatal(1, "ld_out data mismatch: expected 0x%08h got 0x%08h",
                   expected_data, ld_out_data[31:0]);
          end
          seen = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!seen) begin : timeout_out
        $fatal(1, "ld_out timeout (tag=%0d data=0x%08h)", expected_tag, expected_data);
      end
    end
  endtask

  task automatic wait_lddone(input logic expected_tag);
    integer iter_var0;
    logic seen;
    begin : wait_done
      iter_var0 = 0;
      seen = 1'b0;
      if (lddone_valid) begin : got_done_now
        if (lddone_data !== expected_tag) begin : bad_tag_now
          $fatal(1, "lddone tag mismatch: expected %0d got %0d",
                 expected_tag, lddone_data);
        end
        seen = 1'b1;
      end
      while (iter_var0 < 120 && !seen) begin : loop
        @(posedge clk);
        if (lddone_valid) begin : got_done
          if (lddone_data !== expected_tag) begin : bad_tag
            $fatal(1, "lddone tag mismatch: expected %0d got %0d",
                   expected_tag, lddone_data);
          end
          seen = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!seen) begin : timeout_done
        $fatal(1, "lddone timeout for tag %0d", expected_tag);
      end
    end
  endtask

  initial begin : main
    integer pass_count;
    integer iter_var0;
    pass_count = 0;

    rst_n = 1'b0;

    ld_addr_valid = 1'b0;
    ld_addr_data  = '0;
    ld_ctrl_valid = 1'b0;
    ld_ctrl_data  = 1'b0;

    st_addr_valid = 1'b0;
    st_addr_data  = '0;
    st_data_valid = 1'b0;
    st_data_data  = '0;
    st_ctrl_valid = 1'b0;
    st_ctrl_data  = 1'b0;

    ld_out_ready  = 1'b1;
    lddone_ready  = 1'b1;
    stdone_ready  = 1'b1;

    // Region 0: valid=1, start_tag=0, end_tag=2 (half-open [0,2)), addr_offset=0
    // New layout: bits[63:0]=offset, bits[65:64]=end_tag(2b), bit[66]=start_tag(1b), bit[67]=valid
    m0_cfg_data = '0;
    m0_cfg_data[67]    = 1'b1;  // valid
    m0_cfg_data[65:64] = 2'd2;  // end_tag = 2 (half-open [0,2) covers tags 0 and 1)

    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    if (error_valid !== 1'b0) begin : check_reset
      $fatal(1, "error_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    // Tag 0 round-trip.
    drive_store(1'b0, 64'd5, 32'h0000_1122);
    wait_stdone(1'b0);
    drive_ld_addr(1'b0, 64'd5);
    drive_ld_ctrl(1'b0);
    wait_ld_out(1'b0, 32'h0000_1122);
    wait_lddone(1'b0);
    pass_count = pass_count + 1;

    // Tag 1 round-trip.
    drive_store(1'b1, 64'd6, 32'h0000_3344);
    wait_stdone(1'b1);
    drive_ld_addr(1'b1, 64'd6);
    drive_ld_ctrl(1'b1);
    wait_ld_out(1'b1, 32'h0000_3344);
    wait_lddone(1'b1);
    pass_count = pass_count + 1;

    // Mismatched ctrl tag should not consume pending tag-1 address.
    drive_ld_addr(1'b1, 64'd6);
    drive_ld_ctrl(1'b0);

    iter_var0 = 0;
    while (iter_var0 < 10) begin : check_no_early_output
      @(posedge clk);
      if (ld_out_valid) begin : early_out
        $fatal(1, "ld_out should stay low for mismatched ctrl tag");
      end
      iter_var0 = iter_var0 + 1;
    end

    // Matching ctrl tag should release pending load.
    drive_ld_ctrl(1'b1);
    wait_ld_out(1'b1, 32'h0000_3344);
    wait_lddone(1'b1);
    pass_count = pass_count + 1;

    if (error_valid !== 1'b0) begin : check_no_error
      $fatal(1, "unexpected error: code=%0d", error_code);
    end

    $display("PASS: tb_memory_tagged_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : timeout
    #400000;
    $fatal(1, "TIMEOUT");
  end

endmodule
