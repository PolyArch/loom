//===-- tb_fabric_temporal_sw_stress.sv - Temporal switch stress test -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_fabric_temporal_sw_stress;

  parameter int NUM_INPUTS      = 3;
  parameter int NUM_OUTPUTS     = 2;
  parameter int DATA_WIDTH      = 16;
  parameter int TAG_WIDTH       = 3;
  parameter int NUM_ROUTE_TABLE = 4;
  parameter int NUM_CYCLES      = 360;
  parameter int SEED            = 32'h2468_ACE1;

  localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH;
  localparam int SAFE_PW = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1;
  localparam int NUM_CONNECTED = NUM_INPUTS * NUM_OUTPUTS;
  localparam int ENTRY_WIDTH = 1 + TAG_WIDTH + NUM_CONNECTED;
  localparam int CONFIG_WIDTH = NUM_ROUTE_TABLE * ENTRY_WIDTH;

  logic clk;
  logic rst_n;

  logic [NUM_INPUTS-1:0]              in_valid;
  logic [NUM_INPUTS-1:0]              in_ready;
  logic [NUM_INPUTS*SAFE_PW-1:0]      in_data;

  logic [NUM_OUTPUTS-1:0]             out_valid;
  logic [NUM_OUTPUTS-1:0]             out_ready;
  logic [NUM_OUTPUTS*SAFE_PW-1:0]     out_data;

  logic [CONFIG_WIDTH-1:0] cfg_data;

  logic        error_valid;
  logic [15:0] error_code;

  fabric_temporal_sw #(
    .NUM_INPUTS     (NUM_INPUTS),
    .NUM_OUTPUTS    (NUM_OUTPUTS),
    .DATA_WIDTH     (DATA_WIDTH),
    .TAG_WIDTH      (TAG_WIDTH),
    .NUM_ROUTE_TABLE(NUM_ROUTE_TABLE)
  ) dut (
    .clk        (clk),
    .rst_n      (rst_n),
    .in_valid   (in_valid),
    .in_ready   (in_ready),
    .in_data    (in_data),
    .out_valid  (out_valid),
    .out_ready  (out_ready),
    .out_data   (out_data),
    .cfg_data   (cfg_data),
    .error_valid(error_valid),
    .error_code (error_code)
  );

  initial begin : clk_gen
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_fabric_temporal_sw_stress);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_fabric_temporal_sw_stress, "+mda");
  end
`endif

  function automatic int lcg_next(input int state);
    lcg_next = state * 1664525 + 1013904223;
  endfunction

  function automatic logic [ENTRY_WIDTH-1:0] pack_entry(
      input logic valid,
      input logic [TAG_WIDTH-1:0] tag,
      input logic [NUM_CONNECTED-1:0] routes
  );
    logic [ENTRY_WIDTH-1:0] entry;
    begin : pack
      entry = '0;
      entry[0] = valid;
      entry[1 +: TAG_WIDTH] = tag;
      entry[1 + TAG_WIDTH +: NUM_CONNECTED] = routes;
      pack_entry = entry;
    end
  endfunction

  function automatic logic has_route(
      input logic [TAG_WIDTH-1:0] tag,
      input int in_idx,
      input int out_idx
  );
    begin : route_decode
      has_route = 1'b0;
      case (tag)
        3'd1: begin : tag1
          if (out_idx == 0 && in_idx == 0)
            has_route = 1'b1;
          else if (out_idx == 1 && in_idx == 1)
            has_route = 1'b1;
        end
        3'd2: begin : tag2
          if (out_idx == 0 && in_idx == 1)
            has_route = 1'b1;
          else if (out_idx == 1 && in_idx == 2)
            has_route = 1'b1;
        end
        3'd3: begin : tag3
          if (in_idx == 0 && (out_idx == 0 || out_idx == 1))
            has_route = 1'b1;
        end
        default: begin : def
          has_route = 1'b0;
        end
      endcase
    end
  endfunction

  initial begin : main
    int pass_count;
    int handshake_count;
    int rng;
    int iter_var0;
    int iter_var1;
    int iter_var2;
    int cand_count;
    int contender_out0;
    int contender_out1;
    logic [TAG_WIDTH-1:0] in_tag [NUM_INPUTS];
    logic data_match;

    pass_count = 0;
    handshake_count = 0;
    rng = SEED;

    rst_n = 1'b0;
    in_valid = '0;
    in_data = '0;
    out_ready = '0;
    cfg_data = '0;

    // Slots:
    // tag1: out0<-in0, out1<-in1
    // tag2: out0<-in1, out1<-in2
    // tag3: out0<-in0, out1<-in0 (broadcast)
    cfg_data[0 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b1, 3'd1, 6'b01_0001);
    cfg_data[1 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b1, 3'd2, 6'b10_0010);
    cfg_data[2 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b1, 3'd3, 6'b00_1001);
    cfg_data[3 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b0, 3'd0, '0);

    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    if (error_valid !== 1'b0) begin : check_reset
      $fatal(1, "error_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    for (iter_var0 = 0; iter_var0 < NUM_CYCLES; iter_var0 = iter_var0 + 1) begin : stress_loop
      @(negedge clk);

      for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1 = iter_var1 + 1) begin : rand_ready
        rng = lcg_next(rng);
        out_ready[iter_var1] = rng[0];
      end

      // Generate routed-only tags per input to avoid no-match/unrouted errors.
      // in0 -> {1,3}, in1 -> {1,2}, in2 -> {2}
      for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1 = iter_var1 + 1) begin : rand_inputs
        rng = lcg_next(rng);
        in_valid[iter_var1] = rng[0];

        rng = lcg_next(rng);
        if (iter_var1 == 0) begin : in0_tag
          in_tag[iter_var1] = (rng[0]) ? 3'd1 : 3'd3;
        end else if (iter_var1 == 1) begin : in1_tag
          in_tag[iter_var1] = (rng[0]) ? 3'd1 : 3'd2;
        end else begin : in2_tag
          in_tag[iter_var1] = 3'd2;
        end

        rng = lcg_next(rng);
        in_data[iter_var1 * SAFE_PW +: SAFE_PW] = {in_tag[iter_var1], DATA_WIDTH'(rng)};
      end

      #1;

      if (error_valid !== 1'b0) begin : unexpected_err
        $fatal(1, "unexpected error in stress loop: code=%0d cycle=%0d",
               error_code, iter_var0);
      end

      // Invariant: any asserted output must match at least one valid routed input.
      for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1 = iter_var1 + 1) begin : check_out
        cand_count = 0;
        data_match = 1'b0;
        for (iter_var2 = 0; iter_var2 < NUM_INPUTS; iter_var2 = iter_var2 + 1) begin : scan_in
          if (in_valid[iter_var2] && has_route(in_tag[iter_var2], iter_var2, iter_var1)) begin : cand
            cand_count = cand_count + 1;
            if (out_data[iter_var1 * SAFE_PW +: SAFE_PW] == in_data[iter_var2 * SAFE_PW +: SAFE_PW])
              data_match = 1'b1;
          end
        end
        if (out_valid[iter_var1]) begin : must_match
          if (cand_count == 0 || !data_match) begin : bad
            $fatal(1, "out%0d has invalid source at cycle %0d", iter_var1, iter_var0);
          end
        end
      end

      // Broadcast backpressure invariant for tag=3 on in0.
      if (in_valid[0] && in_tag[0] == 3'd3 && (!out_ready[0] || !out_ready[1])) begin : bcast_bp
        if (in_ready[0] !== 1'b0)
          $fatal(1, "broadcast input should be backpressured when one target not ready");
      end

      // If no contention and both targets ready, broadcast should advance.
      contender_out0 = (in_valid[1] && in_tag[1] == 3'd2) ? 1 : 0;
      contender_out1 = (in_valid[1] && in_tag[1] == 3'd1) ||
                       (in_valid[2] && in_tag[2] == 3'd2);
      if (in_valid[0] && in_tag[0] == 3'd3 && out_ready[0] && out_ready[1] &&
          contender_out0 == 0 && contender_out1 == 0) begin : bcast_progress
        if (in_ready[0] !== 1'b1)
          $fatal(1, "broadcast input should be ready without contention");
      end

      for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1 = iter_var1 + 1) begin : count_hs
        if (out_valid[iter_var1] && out_ready[iter_var1])
          handshake_count = handshake_count + 1;
      end

      @(posedge clk);
    end

    if (handshake_count < 20) begin : too_few
      $fatal(1, "stress produced too few handshakes (%0d)", handshake_count);
    end
    pass_count = pass_count + 1;

    // Explicit duplicate-tag CFG error.
    rst_n = 1'b0;
    in_valid = '0;
    out_ready = '0;
    cfg_data = '0;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    cfg_data[0 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b1, 3'd5, 6'b01_0001);
    cfg_data[1 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b1, 3'd5, 6'b10_0010);
    cfg_data[2 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b0, 3'd0, '0);
    cfg_data[3 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b0, 3'd0, '0);

    @(posedge clk);
    @(posedge clk);

    if (error_valid !== 1'b1 || error_code !== CFG_TEMPORAL_SW_DUP_TAG) begin : dup_check
      $fatal(1, "expected CFG_TEMPORAL_SW_DUP_TAG, got valid=%0b code=%0d",
             error_valid, error_code);
    end
    pass_count = pass_count + 1;

    // ---- Test 2a: deterministic RR pointer sequence ----
    // Two slots: tag=1 routes in0->out0, tag=2 routes in1->out0.
    // Both in0 and in1 contend for out0. RR starts at 0 after reset.
    rst_n = 1'b0;
    in_valid = '0;
    in_data = '0;
    out_ready = '0;
    cfg_data = '0;
    // tag=1: in0->out0 -> routes = 6'b00_0001
    cfg_data[0 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b1, 3'd1, 6'b00_0001);
    // tag=2: in1->out0 -> routes = 6'b00_0010
    cfg_data[1 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b1, 3'd2, 6'b00_0010);
    cfg_data[2 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b0, 3'd0, '0);
    cfg_data[3 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b0, 3'd0, '0);
    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Drive both in0 (tag=1) and in1 (tag=2) valid simultaneously, out_ready=1.
    // RR starts at 0: in0 wins first, pointer advances to 1, then in1 wins, etc.
    for (iter_var0 = 0; iter_var0 < 8; iter_var0 = iter_var0 + 1) begin : rr_check
      @(negedge clk);
      in_valid[0] = 1'b1;
      in_data[0 * SAFE_PW +: SAFE_PW] = {3'd1, DATA_WIDTH'(iter_var0 * 2 + 100)};
      in_valid[1] = 1'b1;
      in_data[1 * SAFE_PW +: SAFE_PW] = {3'd2, DATA_WIDTH'(iter_var0 * 2 + 200)};
      in_valid[2] = 1'b0;
      out_ready = '1;
      #1;

      if (!out_valid[0]) begin : rr_no_valid
        $fatal(1, "RR test: out0 should be valid at cycle %0d", iter_var0);
      end
      // Even cycles: in0 wins (RR ptr starts at 0). Odd cycles: in1 wins.
      if ((iter_var0 % 2) == 0) begin : expect_in0
        if (out_data[0 * SAFE_PW +: SAFE_PW] !== {3'd1, DATA_WIDTH'(iter_var0 * 2 + 100)}) begin : bad_rr_data
          $fatal(1, "RR test: expected in0 data at cycle %0d", iter_var0);
        end
      end else begin : expect_in1
        if (out_data[0 * SAFE_PW +: SAFE_PW] !== {3'd2, DATA_WIDTH'(iter_var0 * 2 + 200)}) begin : bad_rr_data
          $fatal(1, "RR test: expected in1 data at cycle %0d", iter_var0);
        end
      end
      @(posedge clk);
    end
    pass_count = pass_count + 1;

    // ---- Test 2b: RR pointer idle hold ----
    // After 8 cycles above, pointer should be at 0 (8 handshakes: 0->1->0->...->0).
    // Insert an idle cycle then verify in0 wins (pointer did not advance).
    @(negedge clk);
    in_valid = '0;
    out_ready = '1;
    @(posedge clk);
    // Now drive both again.
    @(negedge clk);
    in_valid[0] = 1'b1;
    in_data[0 * SAFE_PW +: SAFE_PW] = {3'd1, DATA_WIDTH'(16'hBB00)};
    in_valid[1] = 1'b1;
    in_data[1 * SAFE_PW +: SAFE_PW] = {3'd2, DATA_WIDTH'(16'hCC00)};
    in_valid[2] = 1'b0;
    out_ready = '1;
    #1;
    if (out_data[0 * SAFE_PW +: SAFE_PW] !== {3'd1, DATA_WIDTH'(16'hBB00)}) begin : idle_hold_check
      $fatal(1, "RR idle hold: expected in0 to win after idle (pointer did not advance)");
    end
    @(posedge clk);
    in_valid = '0;
    pass_count = pass_count + 1;

    // ---- Test 2c: multi-error precedence (dup_tag + per-slot fan-in) ----
    // Per-slot fan-in (code 4) must win over dup_tag (code 5).
    rst_n = 1'b0;
    in_valid = '0;
    out_ready = '0;
    cfg_data = '0;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Slot 0: tag=5, in0 AND in1 both route to out0 -> per-slot fan-in (code 4).
    // routes = 6'b00_0011 (out0<-in0 + out0<-in1)
    cfg_data[0 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b1, 3'd5, 6'b00_0011);
    // Slot 1: tag=5 (duplicate) -> dup_tag (code 5).
    cfg_data[1 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b1, 3'd5, 6'b00_0100);
    cfg_data[2 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b0, 3'd0, '0);
    cfg_data[3 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b0, 3'd0, '0);
    @(posedge clk);
    @(posedge clk);

    if (error_valid !== 1'b1 || error_code !== CFG_TEMPORAL_SW_ROUTE_SAME_TAG_INPUTS_TO_SAME_OUTPUT) begin : multi_err_tsw_check
      $fatal(1, "multi-error TSW: expected code %0d, got valid=%0b code=%0d",
             CFG_TEMPORAL_SW_ROUTE_SAME_TAG_INPUTS_TO_SAME_OUTPUT, error_valid, error_code);
    end
    pass_count = pass_count + 1;

    // ---- Test 2d: RT_TEMPORAL_SW_UNROUTED_INPUT ----
    // One valid slot: tag=1 routes in0->out0 only.
    // Drive in1 valid with tag=1 -> in1 matches but has no route -> code 263.
    rst_n = 1'b0;
    in_valid = '0;
    out_ready = '0;
    cfg_data = '0;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // tag=1: in0->out0 only, routes = 6'b00_0001
    cfg_data[0 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b1, 3'd1, 6'b00_0001);
    cfg_data[1 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b0, 3'd0, '0);
    cfg_data[2 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b0, 3'd0, '0);
    cfg_data[3 * ENTRY_WIDTH +: ENTRY_WIDTH] = pack_entry(1'b0, 3'd0, '0);

    @(negedge clk);
    // in1 sends tag=1 but slot for tag=1 does not route in1 -> unrouted input.
    in_valid[1] = 1'b1;
    in_data[1 * SAFE_PW +: SAFE_PW] = {3'd1, DATA_WIDTH'(16'hFFFF)};
    out_ready = '1;
    @(posedge clk);
    @(posedge clk);

    if (error_valid !== 1'b1 || error_code !== RT_TEMPORAL_SW_UNROUTED_INPUT) begin : unrouted_check
      $fatal(1, "unrouted input: expected code %0d, got valid=%0b code=%0d",
             RT_TEMPORAL_SW_UNROUTED_INPUT, error_valid, error_code);
    end
    pass_count = pass_count + 1;

    $display("PASS: tb_fabric_temporal_sw_stress (%0d checks, %0d handshakes)",
             pass_count, handshake_count);
    $finish;
  end

  initial begin : timeout
    #350000;
    $fatal(1, "TIMEOUT");
  end

endmodule
