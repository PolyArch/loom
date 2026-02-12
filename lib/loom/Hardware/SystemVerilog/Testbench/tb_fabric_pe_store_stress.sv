//===-- tb_fabric_pe_store_stress.sv - Store PE stress test ----*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_fabric_pe_store_stress;

  parameter int ELEM_WIDTH  = 32;
  parameter int ADDR_WIDTH  = 16;
  parameter int TAG_WIDTH   = 2;
  parameter int QUEUE_DEPTH = 4;
  parameter int NUM_CYCLES  = 320;
  parameter int SEED        = 32'h25C8_07D1;

  localparam int TAG_COUNT = (1 << TAG_WIDTH);
  localparam int TAG_MASK = TAG_COUNT - 1;
  localparam int ADDR_PW = ADDR_WIDTH + TAG_WIDTH;
  localparam int ELEM_PW = ELEM_WIDTH + TAG_WIDTH;

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
  logic [TAG_WIDTH-1:0] in2_data;

  logic               out0_valid;
  logic               out0_ready;
  logic [ADDR_PW-1:0] out0_data;

  logic               out1_valid;
  logic               out1_ready;
  logic [ELEM_PW-1:0] out1_data;

  logic [0:0] cfg_data;

  fabric_pe_store #(
    .ELEM_WIDTH(ELEM_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH),
    .TAG_WIDTH(TAG_WIDTH),
    .HW_TYPE(1),
    .QUEUE_DEPTH(QUEUE_DEPTH)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .in0_valid(in0_valid),
    .in0_ready(in0_ready),
    .in0_data(in0_data),
    .in1_valid(in1_valid),
    .in1_ready(in1_ready),
    .in1_data(in1_data),
    .in2_valid(in2_valid),
    .in2_ready(in2_ready),
    .in2_data(in2_data),
    .out0_valid(out0_valid),
    .out0_ready(out0_ready),
    .out0_data(out0_data),
    .out1_valid(out1_valid),
    .out1_ready(out1_ready),
    .out1_data(out1_data),
    .cfg_data(cfg_data)
  );

  initial begin : clk_gen
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_fabric_pe_store_stress);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_fabric_pe_store_stress, "+mda");
  end
`endif

  function automatic int lcg_next(input int state);
    lcg_next = state * 1103515245 + 12345;
  endfunction

  function automatic [ADDR_PW-1:0] pack_addr(input int tag, input int addr_value);
    pack_addr = {TAG_WIDTH'(tag), ADDR_WIDTH'(addr_value)};
  endfunction

  function automatic [ELEM_PW-1:0] pack_data(input int tag, input int data_value);
    pack_data = {TAG_WIDTH'(tag), ELEM_WIDTH'(data_value)};
  endfunction

  initial begin : main
    int pass_count;
    int rng;
    int addr_occ;
    int data_occ;
    int ctrl_occ;
    int iter_var0;
    int iter_var1;
    int tag_sel;
    int match_tag;
    int out_hs_count;
    int addr_push_count;
    int data_push_count;
    int ctrl_push_count;
    logic addr_expect_full;
    logic data_expect_full;
    logic ctrl_expect_full;
    logic addr_push;
    logic data_push;
    logic ctrl_push;
    logic out0_fire;
    logic out1_fire;
    logic atomic_fire;
    logic [ADDR_WIDTH-1:0] pending_addr_value [TAG_COUNT];
    logic [ELEM_WIDTH-1:0] pending_data_value [TAG_COUNT];
    logic [TAG_COUNT-1:0] pending_addr_valid;
    logic [TAG_COUNT-1:0] pending_data_valid;
    logic [TAG_COUNT-1:0] pending_ctrl_valid;

    pass_count = 0;
    rng = SEED;
    addr_occ = 0;
    data_occ = 0;
    ctrl_occ = 0;
    out_hs_count = 0;
    addr_push_count = 0;
    data_push_count = 0;
    ctrl_push_count = 0;
    pending_addr_valid = '0;
    pending_data_valid = '0;
    pending_ctrl_valid = '0;
    for (iter_var0 = 0; iter_var0 < TAG_COUNT; iter_var0 = iter_var0 + 1) begin : init_arr
      pending_addr_value[iter_var0] = '0;
      pending_data_value[iter_var0] = '0;
    end

    rst_n = 1'b0;
    in0_valid = 1'b0;
    in0_data = '0;
    in1_valid = 1'b0;
    in1_data = '0;
    in2_valid = 1'b0;
    in2_data = '0;
    out0_ready = 1'b0;
    out1_ready = 1'b0;
    cfg_data = '0;

    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Address queue backpressure check.
    for (iter_var0 = 0; iter_var0 < QUEUE_DEPTH; iter_var0 = iter_var0 + 1) begin : fill_addr_q
      @(negedge clk);
      in0_valid = 1'b1;
      in0_data = pack_addr(iter_var0 % TAG_COUNT, 32'h0200 + iter_var0);
      in1_valid = 1'b0;
      in2_valid = 1'b0;
      out0_ready = 1'b0;
      out1_ready = 1'b0;
      #1;
      if (in0_ready !== 1'b1) begin : addr_fill_ready
        $fatal(1, "address queue should accept entry %0d", iter_var0);
      end
      @(posedge clk);
    end
    @(negedge clk);
    in0_valid = 1'b1;
    in0_data = pack_addr(0, 32'h0ABC);
    #1;
    if (in0_ready !== 1'b0) begin : addr_full_backpressure
      $fatal(1, "address queue backpressure missing at full depth");
    end
    pass_count = pass_count + 1;

    rst_n = 1'b0;
    in0_valid = 1'b0;
    in1_valid = 1'b0;
    in2_valid = 1'b0;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Data queue backpressure check.
    for (iter_var0 = 0; iter_var0 < QUEUE_DEPTH; iter_var0 = iter_var0 + 1) begin : fill_data_q
      @(negedge clk);
      in0_valid = 1'b0;
      in1_valid = 1'b1;
      in1_data = pack_data(iter_var0 % TAG_COUNT, 32'h3000 + iter_var0);
      in2_valid = 1'b0;
      out0_ready = 1'b0;
      out1_ready = 1'b0;
      #1;
      if (in1_ready !== 1'b1) begin : data_fill_ready
        $fatal(1, "data queue should accept entry %0d", iter_var0);
      end
      @(posedge clk);
    end
    @(negedge clk);
    in1_valid = 1'b1;
    in1_data = pack_data(0, 32'h0DEF);
    #1;
    if (in1_ready !== 1'b0) begin : data_full_backpressure
      $fatal(1, "data queue backpressure missing at full depth");
    end
    pass_count = pass_count + 1;

    rst_n = 1'b0;
    in0_valid = 1'b0;
    in1_valid = 1'b0;
    in2_valid = 1'b0;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Control queue backpressure check.
    for (iter_var0 = 0; iter_var0 < QUEUE_DEPTH; iter_var0 = iter_var0 + 1) begin : fill_ctrl_q
      @(negedge clk);
      in0_valid = 1'b0;
      in1_valid = 1'b0;
      in2_valid = 1'b1;
      in2_data = TAG_WIDTH'(iter_var0 % TAG_COUNT);
      out0_ready = 1'b0;
      out1_ready = 1'b0;
      #1;
      if (in2_ready !== 1'b1) begin : ctrl_fill_ready
        $fatal(1, "control queue should accept entry %0d", iter_var0);
      end
      @(posedge clk);
    end
    @(negedge clk);
    in2_valid = 1'b1;
    in2_data = TAG_WIDTH'(0);
    #1;
    if (in2_ready !== 1'b0) begin : ctrl_full_backpressure
      $fatal(1, "control queue backpressure missing at full depth");
    end
    pass_count = pass_count + 1;

    // Fresh randomized phase.
    rst_n = 1'b0;
    in0_valid = 1'b0;
    in1_valid = 1'b0;
    in2_valid = 1'b0;
    out0_ready = 1'b0;
    out1_ready = 1'b0;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    addr_occ = 0;
    data_occ = 0;
    ctrl_occ = 0;
    pending_addr_valid = '0;
    pending_data_valid = '0;
    pending_ctrl_valid = '0;

    for (iter_var0 = 0; iter_var0 < NUM_CYCLES; iter_var0 = iter_var0 + 1) begin : stress_loop
      match_tag = -1;
      for (iter_var1 = 0; iter_var1 < TAG_COUNT; iter_var1 = iter_var1 + 1) begin : find_match
        if (pending_addr_valid[iter_var1] &&
            pending_data_valid[iter_var1] &&
            pending_ctrl_valid[iter_var1]) begin : hit
          match_tag = iter_var1;
        end
      end

      @(negedge clk);
      rng = lcg_next(rng);
      out0_ready = iter_var0[0] || rng[0];
      rng = lcg_next(rng);
      out1_ready = iter_var0[1] || rng[1];

      in0_valid = 1'b0;
      in0_data = '0;
      in1_valid = 1'b0;
      in1_data = '0;
      in2_valid = 1'b0;
      in2_data = '0;

      if (match_tag < 0) begin : gen_inputs
        rng = lcg_next(rng);
        tag_sel = (rng >> 8) & TAG_MASK;

        if (addr_occ < QUEUE_DEPTH && !pending_addr_valid[tag_sel]) begin : drive_addr_first
          in0_valid = 1'b1;
          in0_data = pack_addr(tag_sel, (32'h5000 + iter_var0) ^ (tag_sel << 2));
        end else if (data_occ < QUEUE_DEPTH && !pending_data_valid[tag_sel]) begin : drive_data_second
          in1_valid = 1'b1;
          in1_data = pack_data(tag_sel, (32'h7000_0000 + iter_var0) ^ (tag_sel << 8));
        end else if (ctrl_occ < QUEUE_DEPTH && !pending_ctrl_valid[tag_sel]) begin : drive_ctrl_third
          in2_valid = 1'b1;
          in2_data = TAG_WIDTH'(tag_sel);
        end
      end

      #1;

      addr_expect_full = (addr_occ >= QUEUE_DEPTH);
      data_expect_full = (data_occ >= QUEUE_DEPTH);
      ctrl_expect_full = (ctrl_occ >= QUEUE_DEPTH);
      if (in0_ready !== !addr_expect_full) begin : check_addr_ready
        $fatal(1, "in0_ready mismatch at cycle=%0d occ=%0d", iter_var0, addr_occ);
      end
      if (in1_ready !== !data_expect_full) begin : check_data_ready
        $fatal(1, "in1_ready mismatch at cycle=%0d occ=%0d", iter_var0, data_occ);
      end
      if (in2_ready !== !ctrl_expect_full) begin : check_ctrl_ready
        $fatal(1, "in2_ready mismatch at cycle=%0d occ=%0d", iter_var0, ctrl_occ);
      end

      if ((match_tag >= 0 && out0_valid !== out1_ready) ||
          (match_tag < 0 && out0_valid !== 1'b0)) begin : check_out0_valid
        $fatal(1, "out0_valid mismatch at cycle=%0d match_tag=%0d", iter_var0, match_tag);
      end
      if ((match_tag >= 0 && out1_valid !== out0_ready) ||
          (match_tag < 0 && out1_valid !== 1'b0)) begin : check_out1_valid
        $fatal(1, "out1_valid mismatch at cycle=%0d match_tag=%0d", iter_var0, match_tag);
      end

      if (match_tag >= 0) begin : check_out_data
        if (out0_data[ADDR_WIDTH +: TAG_WIDTH] !== TAG_WIDTH'(match_tag)) begin : bad_addr_tag
          $fatal(1, "out0 tag mismatch at cycle=%0d", iter_var0);
        end
        if (out0_data[ADDR_WIDTH-1:0] !== pending_addr_value[match_tag]) begin : bad_addr_value
          $fatal(1, "out0 address mismatch at cycle=%0d", iter_var0);
        end
        if (out1_data[ELEM_WIDTH +: TAG_WIDTH] !== TAG_WIDTH'(match_tag)) begin : bad_data_tag
          $fatal(1, "out1 tag mismatch at cycle=%0d", iter_var0);
        end
        if (out1_data[ELEM_WIDTH-1:0] !== pending_data_value[match_tag]) begin : bad_data_value
          $fatal(1, "out1 data mismatch at cycle=%0d", iter_var0);
        end
      end

      out0_fire = (out0_valid && out0_ready);
      out1_fire = (out1_valid && out1_ready);
      if (out0_fire !== out1_fire) begin : check_atomic
        $fatal(1, "atomic output handshake violated at cycle=%0d", iter_var0);
      end
      atomic_fire = out0_fire;
      if (atomic_fire && match_tag < 0) begin : check_fire_match
        $fatal(1, "store fired without a match at cycle=%0d", iter_var0);
      end

      addr_push = (in0_valid && in0_ready);
      data_push = (in1_valid && in1_ready);
      ctrl_push = (in2_valid && in2_ready);

      @(posedge clk);

      if (atomic_fire) begin : apply_pop
        pending_addr_valid[match_tag] = 1'b0;
        pending_data_valid[match_tag] = 1'b0;
        pending_ctrl_valid[match_tag] = 1'b0;
        addr_occ = addr_occ - 1;
        data_occ = data_occ - 1;
        ctrl_occ = ctrl_occ - 1;
        out_hs_count = out_hs_count + 1;
      end

      if (addr_push) begin : apply_addr_push
        pending_addr_valid[in0_data[ADDR_WIDTH +: TAG_WIDTH]] = 1'b1;
        pending_addr_value[in0_data[ADDR_WIDTH +: TAG_WIDTH]] = in0_data[ADDR_WIDTH-1:0];
        addr_occ = addr_occ + 1;
        addr_push_count = addr_push_count + 1;
      end

      if (data_push) begin : apply_data_push
        pending_data_valid[in1_data[ELEM_WIDTH +: TAG_WIDTH]] = 1'b1;
        pending_data_value[in1_data[ELEM_WIDTH +: TAG_WIDTH]] = in1_data[ELEM_WIDTH-1:0];
        data_occ = data_occ + 1;
        data_push_count = data_push_count + 1;
      end

      if (ctrl_push) begin : apply_ctrl_push
        pending_ctrl_valid[in2_data] = 1'b1;
        ctrl_occ = ctrl_occ + 1;
        ctrl_push_count = ctrl_push_count + 1;
      end
    end

    if (out_hs_count < 20) begin : check_hs_count
      $fatal(1, "too few matched store handshakes: %0d", out_hs_count);
    end
    if (addr_push_count < 20 || data_push_count < 20 || ctrl_push_count < 20) begin : check_push_counts
      $fatal(1, "too few queue pushes: addr=%0d data=%0d ctrl=%0d",
             addr_push_count, data_push_count, ctrl_push_count);
    end
    pass_count = pass_count + 1;

    $display("PASS: tb_fabric_pe_store_stress (%0d checks, handshakes=%0d)",
             pass_count, out_hs_count);
    $finish;
  end

  initial begin : timeout
    #300000;
    $fatal(1, "TIMEOUT");
  end

endmodule
