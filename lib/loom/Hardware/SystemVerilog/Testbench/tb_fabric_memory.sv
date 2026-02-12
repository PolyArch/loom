//===-- tb_fabric_memory.sv - Parameterized memory test --------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_fabric_memory;
  parameter int ADDR_WIDTH       = 64;
  parameter int ELEM_WIDTH       = 32;
  parameter int TAG_WIDTH        = 0;
  parameter int LD_COUNT         = 1;
  parameter int ST_COUNT         = 1;
  parameter int LSQ_DEPTH        = 4;
  parameter int IS_PRIVATE       = 1;
  parameter int MEM_DEPTH        = 64;
  parameter int DEADLOCK_TIMEOUT = 65535;

  localparam int ADDR_PW      = ADDR_WIDTH + TAG_WIDTH;
  localparam int ELEM_PW      = ELEM_WIDTH + TAG_WIDTH;
  localparam int DONE_PW      = (TAG_WIDTH > 0) ? TAG_WIDTH : 1;
  localparam int SAFE_ADDR_PW = (ADDR_PW > 0) ? ADDR_PW : 1;
  localparam int SAFE_ELEM_PW = (ELEM_PW > 0) ? ELEM_PW : 1;
  localparam int SAFE_TW      = (TAG_WIDTH > 0) ? TAG_WIDTH : 1;

  logic clk, rst_n;

  // Named port signals (match fabric_memory interface)
  logic                     ld_addr_valid, ld_addr_ready;
  logic [SAFE_ADDR_PW-1:0] ld_addr_data;

  logic                     st_addr_valid, st_addr_ready;
  logic [SAFE_ADDR_PW-1:0] st_addr_data;

  logic                     st_data_valid, st_data_ready;
  logic [SAFE_ELEM_PW-1:0] st_data_data;

  logic                     ld_data_valid, ld_data_ready;
  logic [SAFE_ELEM_PW-1:0] ld_data_data;

  logic                     ld_done_valid, ld_done_ready;
  logic [DONE_PW-1:0]      ld_done_data;

  logic                     st_done_valid, st_done_ready;
  logic [DONE_PW-1:0]      st_done_data;

  logic                     memref_valid, memref_ready;

  logic        error_valid;
  logic [15:0] error_code;

  fabric_memory #(
    .ADDR_WIDTH(ADDR_WIDTH),
    .ELEM_WIDTH(ELEM_WIDTH),
    .TAG_WIDTH(TAG_WIDTH),
    .LD_COUNT(LD_COUNT),
    .ST_COUNT(ST_COUNT),
    .LSQ_DEPTH(LSQ_DEPTH),
    .IS_PRIVATE(IS_PRIVATE),
    .MEM_DEPTH(MEM_DEPTH),
    .DEADLOCK_TIMEOUT(DEADLOCK_TIMEOUT)
  ) dut (
    .clk(clk), .rst_n(rst_n),
    .ld_addr_valid(ld_addr_valid), .ld_addr_ready(ld_addr_ready),
    .ld_addr_data(ld_addr_data),
    .st_addr_valid(st_addr_valid), .st_addr_ready(st_addr_ready),
    .st_addr_data(st_addr_data),
    .st_data_valid(st_data_valid), .st_data_ready(st_data_ready),
    .st_data_data(st_data_data),
    .ld_data_valid(ld_data_valid), .ld_data_ready(ld_data_ready),
    .ld_data_data(ld_data_data),
    .ld_done_valid(ld_done_valid), .ld_done_ready(ld_done_ready),
    .ld_done_data(ld_done_data),
    .st_done_valid(st_done_valid), .st_done_ready(st_done_ready),
    .st_done_data(st_done_data),
    .memref_valid(memref_valid), .memref_ready(memref_ready),
    .error_valid(error_valid), .error_code(error_code)
  );

  initial begin : clk_gen
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin : test
    integer pass_count;
    integer iter_var0;
    integer iter_var1;
    integer rand_state;
    integer tag_limit;
    integer done_count;
    logic seen_tag0;
    logic seen_tag1;
    logic [ADDR_WIDTH-1:0] stress_addr_raw;
    logic [ELEM_WIDTH-1:0] stress_data_raw;
    logic [SAFE_ADDR_PW-1:0] stress_addr_payload;
    logic [SAFE_ELEM_PW-1:0] stress_data_payload;
    logic [SAFE_ADDR_PW-1:0] stress_ld_addr_payload;
    logic [DONE_PW-1:0] expected_tag;
    pass_count = 0;
    rst_n = 0;
    ld_addr_valid = 0;
    ld_addr_data  = '0;
    st_addr_valid = 0;
    st_addr_data  = '0;
    st_data_valid = 0;
    st_data_data  = '0;
    ld_data_ready = 1;
    ld_done_ready = 1;
    st_done_ready = 1;
    memref_ready  = 1;

    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // Check 1: no error after reset
    if (error_valid !== 1'b0) begin : check_reset
      $fatal(1, "error_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    // Check 2: store then load - verify round-trip through FIFO-based store queue
    if (ST_COUNT > 0 && LD_COUNT > 0) begin : store_load
      // Store: addr=5, data=0xABCD via single port pair
      st_addr_data = SAFE_ADDR_PW'(5);
      st_addr_valid = 1'b1;
      st_data_data = SAFE_ELEM_PW'(32'hABCD);
      st_data_valid = 1'b1;
      @(posedge clk);
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      // Wait for store done
      iter_var0 = 0;
      while (iter_var0 < 10) begin : wait_stdone
        @(posedge clk);
        iter_var0 = iter_var0 + 1;
        if (st_done_valid) begin : done
          iter_var0 = 10;
        end
      end
      @(posedge clk);

      // Load: addr=5
      ld_addr_data = SAFE_ADDR_PW'(5);
      ld_addr_valid = 1'b1;
      @(posedge clk);
      if (ld_data_valid !== 1'b1) begin : check_ld_valid
        $fatal(1, "load data output should be valid");
      end
      if (ld_data_data[ELEM_WIDTH-1:0] !== ELEM_WIDTH'(32'hABCD)) begin : check_ld_data
        $fatal(1, "load data mismatch: expected 0xABCD, got 0x%0h",
               ld_data_data[ELEM_WIDTH-1:0]);
      end
      pass_count = pass_count + 1;
      ld_addr_valid = 1'b0;
      @(posedge clk);
    end

    // Check 3: no error after normal operation
    if (error_valid !== 1'b0) begin : check_no_err
      $fatal(1, "unexpected error after store/load: code=%0d", error_code);
    end
    pass_count = pass_count + 1;

    // Check 4: RT_MEMORY_TAG_OOB (only testable when TAG_WIDTH > 0 and LD_COUNT > 1)
    if (TAG_WIDTH > 0 && LD_COUNT > 1) begin : tag_oob_test
      rst_n = 0;
      repeat (2) @(posedge clk);
      rst_n = 1;
      @(posedge clk);
      // Send load with tag = LD_COUNT (out of bounds)
      ld_addr_data = SAFE_ADDR_PW'(LD_COUNT) << ADDR_WIDTH;
      ld_addr_valid = 1'b1;
      @(posedge clk);
      @(posedge clk);
      if (error_valid !== 1'b1) begin : check_tag_oob
        $fatal(1, "expected RT_MEMORY_TAG_OOB error");
      end
      if (error_code !== RT_MEMORY_TAG_OOB) begin : check_tag_oob_code
        $fatal(1, "wrong error code for tag OOB: got %0d", error_code);
      end
      pass_count = pass_count + 1;
      ld_addr_valid = 1'b0;
    end

    // Check 5: multi-tag store (when ST_COUNT >= 2 && TAG_WIDTH > 0)
    // Tags are embedded in the upper bits of the single st_addr/st_data ports.
    if (ST_COUNT >= 2 && TAG_WIDTH > 0) begin : cross_tag_test
      rst_n = 0;
      ld_addr_valid = 1'b0;
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      repeat (2) @(posedge clk);
      rst_n = 1;
      @(posedge clk);

      // Store tag=0: addr=10, data=0xDEAD
      st_addr_data = SAFE_ADDR_PW'(10);
      st_data_data = SAFE_ELEM_PW'(32'hDEAD);
      st_addr_valid = 1'b1;
      st_data_valid = 1'b1;
      @(posedge clk);
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      repeat (10) @(posedge clk);

      // Store tag=1: addr=11, data=0xBEEF
      st_addr_data = (SAFE_ADDR_PW'(1) << ADDR_WIDTH) | SAFE_ADDR_PW'(11);
      st_data_data = (SAFE_ELEM_PW'(1) << ELEM_WIDTH) | SAFE_ELEM_PW'(32'hBEEF);
      st_addr_valid = 1'b1;
      st_data_valid = 1'b1;
      @(posedge clk);
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      repeat (10) @(posedge clk);

      // Load addr=10 -> expect 0xDEAD
      ld_addr_data = SAFE_ADDR_PW'(10);
      ld_addr_valid = 1'b1;
      @(posedge clk);
      ld_addr_valid = 1'b0;
      if (ld_data_data[ELEM_WIDTH-1:0] !== ELEM_WIDTH'(32'hDEAD)) begin : check_st0
        $fatal(1, "cross-tag store 0: expected 0xDEAD, got 0x%0h",
               ld_data_data[ELEM_WIDTH-1:0]);
      end
      @(posedge clk);

      // Load addr=11 -> expect 0xBEEF
      ld_addr_data = SAFE_ADDR_PW'(11);
      ld_addr_valid = 1'b1;
      @(posedge clk);
      ld_addr_valid = 1'b0;
      if (ld_data_data[ELEM_WIDTH-1:0] !== ELEM_WIDTH'(32'hBEEF)) begin : check_st1
        $fatal(1, "cross-tag store 1: expected 0xBEEF, got 0x%0h",
               ld_data_data[ELEM_WIDTH-1:0]);
      end
      pass_count = pass_count + 1;
    end

    // Check 6: stdone tag correctness (when ST_COUNT >= 2 && TAG_WIDTH > 0)
    if (ST_COUNT >= 2 && TAG_WIDTH > 0) begin : stdone_tag_test
      rst_n = 0;
      ld_addr_valid = 1'b0;
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      repeat (2) @(posedge clk);
      rst_n = 1;
      @(posedge clk);

      // Store targeting tag=1: addr=20, data=0x1234
      st_addr_data = (SAFE_ADDR_PW'(1) << ADDR_WIDTH) | SAFE_ADDR_PW'(20);
      st_data_data = (SAFE_ELEM_PW'(1) << ELEM_WIDTH) | SAFE_ELEM_PW'(32'h1234);
      st_addr_valid = 1'b1;
      st_data_valid = 1'b1;
      @(posedge clk);
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;

      // Wait for stdone and verify tag
      iter_var0 = 0;
      while (iter_var0 < 10) begin : wait_stdone_tag
        @(posedge clk);
        iter_var0 = iter_var0 + 1;
        if (st_done_valid) begin : got_stdone
          if (st_done_data !== DONE_PW'(1)) begin : bad_tag
            $fatal(1, "stdone tag: expected 1, got %0d", st_done_data);
          end
          iter_var0 = 10;
        end
      end
      pass_count = pass_count + 1;
    end

    // Check 7: sequential same-tag stores (when ST_COUNT >= 2 && TAG_WIDTH > 0)
    // Both stores use tag=0 through the single port, sequentially.
    if (ST_COUNT >= 2 && TAG_WIDTH > 0) begin : same_tag_test
      rst_n = 0;
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      ld_addr_valid = 1'b0;
      repeat (2) @(posedge clk);
      rst_n = 1;
      @(posedge clk);

      // First store tag=0: addr=30, data=0xAAAA
      st_addr_data = SAFE_ADDR_PW'(30);
      st_data_data = SAFE_ELEM_PW'(32'hAAAA);
      st_addr_valid = 1'b1;
      st_data_valid = 1'b1;
      @(posedge clk);
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      repeat (10) @(posedge clk);

      // Second store tag=0: addr=31, data=0xBBBB
      st_addr_data = SAFE_ADDR_PW'(31);
      st_data_data = SAFE_ELEM_PW'(32'hBBBB);
      st_addr_valid = 1'b1;
      st_data_valid = 1'b1;
      @(posedge clk);
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      repeat (10) @(posedge clk);

      // Load addr=30 -> expect 0xAAAA
      ld_addr_data = SAFE_ADDR_PW'(30);
      ld_addr_valid = 1'b1;
      @(posedge clk);
      ld_addr_valid = 1'b0;
      if (ld_data_data[ELEM_WIDTH-1:0] !== ELEM_WIDTH'(32'hAAAA)) begin : check_st0_same
        $fatal(1, "same-tag store 0: expected 0xAAAA, got 0x%0h",
               ld_data_data[ELEM_WIDTH-1:0]);
      end
      @(posedge clk);

      // Load addr=31 -> expect 0xBBBB
      ld_addr_data = SAFE_ADDR_PW'(31);
      ld_addr_valid = 1'b1;
      @(posedge clk);
      ld_addr_valid = 1'b0;
      if (ld_data_data[ELEM_WIDTH-1:0] !== ELEM_WIDTH'(32'hBBBB)) begin : check_st1_same
        $fatal(1, "same-tag store 1: expected 0xBBBB, got 0x%0h",
               ld_data_data[ELEM_WIDTH-1:0]);
      end
      pass_count = pass_count + 1;
    end

    // Check 8: load completion alignment under independent ready toggling
    if (ST_COUNT > 0 && LD_COUNT > 0) begin : ld_align_test
      rst_n = 0;
      ld_addr_valid = 1'b0;
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      ld_data_ready = 1'b1;
      ld_done_ready = 1'b1;
      st_done_ready = 1'b1;
      repeat (2) @(posedge clk);
      rst_n = 1;
      @(posedge clk);

      // Seed addr=6 with known data.
      st_addr_data = SAFE_ADDR_PW'(6);
      st_data_data = SAFE_ELEM_PW'(32'hCAFE);
      st_addr_valid = 1'b1;
      st_data_valid = 1'b1;
      iter_var0 = 0;
      while (iter_var0 < 10 && !(st_addr_ready && st_data_ready)) begin : wait_seed_ready_align
        @(posedge clk);
        iter_var0 = iter_var0 + 1;
      end
      if (!(st_addr_ready && st_data_ready)) begin : check_seed_ready_align
        $fatal(1, "seed store was not accepted in ld_align_test");
      end
      @(posedge clk);
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      iter_var0 = 0;
      while (iter_var0 < 10) begin : wait_seed_done_align
        @(posedge clk);
        iter_var0 = iter_var0 + 1;
        if (st_done_valid) begin : seed_done
          iter_var0 = 10;
        end
      end

      // Keep one completion channel blocked at a time.
      ld_addr_data = SAFE_ADDR_PW'(6);
      ld_addr_valid = 1'b1;
      ld_data_ready = 1'b1;
      ld_done_ready = 1'b0;
      @(posedge clk);
      if (ld_addr_ready !== 1'b0) begin : check_no_addr_fire_a
        $fatal(1, "load accepted while ld_done was not ready");
      end

      ld_data_ready = 1'b0;
      ld_done_ready = 1'b1;
      @(posedge clk);
      if (ld_addr_ready !== 1'b0) begin : check_no_addr_fire_b
        $fatal(1, "load accepted while ld_data was not ready");
      end

      // Release both ready signals: lddata and lddone must assert together.
      ld_data_ready = 1'b1;
      ld_done_ready = 1'b1;
      @(posedge clk);
      if (ld_addr_ready !== 1'b1) begin : check_addr_fire
        $fatal(1, "load should be accepted when both completion channels are ready");
      end
      if (ld_data_valid !== 1'b1 || ld_done_valid !== 1'b1) begin : check_aligned_valid
        $fatal(1, "ld_data_valid and ld_done_valid must assert together");
      end
      if (ld_data_data[ELEM_WIDTH-1:0] !== ELEM_WIDTH'(32'hCAFE)) begin : check_aligned_data
        $fatal(1, "aligned load returned wrong data: 0x%0h", ld_data_data[ELEM_WIDTH-1:0]);
      end
      ld_addr_valid = 1'b0;
      pass_count = pass_count + 1;
    end

    // Check 9: multi-tag stdone events are not dropped
    if (ST_COUNT >= 2 && TAG_WIDTH > 0) begin : stdone_nonloss_test
      rst_n = 0;
      ld_addr_valid = 1'b0;
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      st_done_ready = 1'b0;
      repeat (2) @(posedge clk);
      rst_n = 1;
      @(posedge clk);

      // Queue one complete store for tag 0.
      st_addr_data = SAFE_ADDR_PW'(40);
      st_data_data = SAFE_ELEM_PW'(32'h1111);
      st_addr_valid = 1'b1;
      st_data_valid = 1'b1;
      iter_var0 = 0;
      while (iter_var0 < 10 && !(st_addr_ready && st_data_ready)) begin : wait_nonloss_ready0
        @(posedge clk);
        iter_var0 = iter_var0 + 1;
      end
      if (!(st_addr_ready && st_data_ready)) begin : check_nonloss_ready0
        $fatal(1, "non-loss test tag0 store was not accepted");
      end
      @(posedge clk);
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;

      // Queue one complete store for tag 1.
      st_addr_data = (SAFE_ADDR_PW'(1) << ADDR_WIDTH) | SAFE_ADDR_PW'(41);
      st_data_data = (SAFE_ELEM_PW'(1) << ELEM_WIDTH) | SAFE_ELEM_PW'(32'h2222);
      st_addr_valid = 1'b1;
      st_data_valid = 1'b1;
      iter_var0 = 0;
      while (iter_var0 < 10 && !(st_addr_ready && st_data_ready)) begin : wait_nonloss_ready1
        @(posedge clk);
        iter_var0 = iter_var0 + 1;
      end
      if (!(st_addr_ready && st_data_ready)) begin : check_nonloss_ready1
        $fatal(1, "non-loss test tag1 store was not accepted");
      end
      @(posedge clk);
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;

      done_count = 0;
      seen_tag0 = 1'b0;
      seen_tag1 = 1'b0;
      st_done_ready = 1'b0;
      while (done_count < 2) begin : collect_two_done
        iter_var0 = 0;
        while (iter_var0 < 20 && !st_done_valid) begin : wait_done_token
          @(posedge clk);
          iter_var0 = iter_var0 + 1;
        end
        if (!st_done_valid) begin : missing_done
          $fatal(1, "expected two stdone tokens, saw only %0d", done_count);
        end

        if (st_done_data === DONE_PW'(0)) begin : mark0
          seen_tag0 = 1'b1;
        end else if (st_done_data === DONE_PW'(1)) begin : mark1
          seen_tag1 = 1'b1;
        end else begin : bad_tag
          $fatal(1, "unexpected stdone tag %0d in non-loss test", st_done_data);
        end
        done_count = done_count + 1;

        st_done_ready = 1'b1;
        @(posedge clk);
        st_done_ready = 1'b0;
        @(posedge clk);
      end
      if (done_count != 2 || !seen_tag0 || !seen_tag1) begin : check_nonloss
        $fatal(1, "expected two stdone events (tags 0 and 1), got count=%0d t0=%0b t1=%0b",
               done_count, seen_tag0, seen_tag1);
      end
      pass_count = pass_count + 1;
    end

    // Check 10: randomized store/load round-trip stress
    if (ST_COUNT > 0 && LD_COUNT > 0) begin : random_roundtrip_stress
      rst_n = 0;
      ld_addr_valid = 1'b0;
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      ld_data_ready = 1'b1;
      ld_done_ready = 1'b1;
      st_done_ready = 1'b1;
      repeat (2) @(posedge clk);
      rst_n = 1;
      @(posedge clk);

      rand_state = 32'h31415926;
      if (TAG_WIDTH > 0)
        tag_limit = (LD_COUNT < ST_COUNT) ? LD_COUNT : ST_COUNT;
      else
        tag_limit = 1;
      if (tag_limit < 1)
        tag_limit = 1;

      for (iter_var1 = 0; iter_var1 < 24; iter_var1 = iter_var1 + 1) begin : per_txn
        rand_state = rand_state * 1664525 + 1013904223;
        stress_addr_raw = ADDR_WIDTH'((ADDR_WIDTH'(rand_state) & ADDR_WIDTH'(32'h7FFF_FFFF)) % MEM_DEPTH);
        rand_state = rand_state * 1664525 + 1013904223;
        stress_data_raw = ELEM_WIDTH'(rand_state);

        if (TAG_WIDTH > 0) begin : tagged_payload
          rand_state = rand_state * 1664525 + 1013904223;
          expected_tag = DONE_PW'((rand_state & 32'h7FFF_FFFF) % tag_limit);
          stress_addr_payload =
              (SAFE_ADDR_PW'(expected_tag) << ADDR_WIDTH) | SAFE_ADDR_PW'(stress_addr_raw);
          stress_data_payload =
              (SAFE_ELEM_PW'(expected_tag) << ELEM_WIDTH) | SAFE_ELEM_PW'(stress_data_raw);
          stress_ld_addr_payload =
              (SAFE_ADDR_PW'(expected_tag) << ADDR_WIDTH) | SAFE_ADDR_PW'(stress_addr_raw);
        end else begin : native_payload
          expected_tag = DONE_PW'(0);
          stress_addr_payload = SAFE_ADDR_PW'(stress_addr_raw);
          stress_data_payload = SAFE_ELEM_PW'(stress_data_raw);
          stress_ld_addr_payload = SAFE_ADDR_PW'(stress_addr_raw);
        end

        // Store request (addr+data pair)
        st_addr_data = stress_addr_payload;
        st_data_data = stress_data_payload;
        st_addr_valid = 1'b1;
        st_data_valid = 1'b1;
        iter_var0 = 0;
        while (iter_var0 < 20 && !(st_addr_ready && st_data_ready)) begin : wait_store_accept
          @(posedge clk);
          iter_var0 = iter_var0 + 1;
        end
        if (!(st_addr_ready && st_data_ready)) begin : check_store_accept
          $fatal(1, "random stress store not accepted (txn %0d)", iter_var1);
        end
        @(posedge clk);
        st_addr_valid = 1'b0;
        st_data_valid = 1'b0;

        // Wait for matching stdone tag.
        seen_tag0 = 1'b0;
        iter_var0 = 0;
        if (TAG_WIDTH > 0) begin : precheck_stdone_tagged
          if (st_done_valid && (st_done_data === expected_tag))
            seen_tag0 = 1'b1;
        end else begin : precheck_stdone_native
          if (st_done_valid)
            seen_tag0 = 1'b1;
        end
        while (iter_var0 < 40 && !seen_tag0) begin : wait_stdone_match
          @(posedge clk);
          if (TAG_WIDTH > 0) begin : chk_stdone_tagged
            if (st_done_valid && (st_done_data === expected_tag))
              seen_tag0 = 1'b1;
          end else begin : chk_stdone_native
            if (st_done_valid)
              seen_tag0 = 1'b1;
          end
          iter_var0 = iter_var0 + 1;
        end
        if (!seen_tag0) begin : check_stdone_match
          $fatal(1, "random stress missing matching stdone (txn %0d, tag %0d)",
                 iter_var1, expected_tag);
        end

        // Load same address/tag and verify aligned completion data.
        ld_addr_data = stress_ld_addr_payload;
        ld_addr_valid = 1'b1;
        seen_tag1 = 1'b0;
        if (TAG_WIDTH > 0) begin : tagged_load_roundtrip
          iter_var0 = 0;
          while (iter_var0 < 20 && !ld_addr_ready) begin : wait_load_accept
            @(posedge clk);
            iter_var0 = iter_var0 + 1;
          end
          if (!ld_addr_ready) begin : check_load_accept
            $fatal(1, "random stress load not accepted (txn %0d)", iter_var1);
          end

          if (ld_data_valid && ld_done_valid) begin : precheck_accept_edge
            if (ld_data_data[ELEM_WIDTH-1:0] !== stress_data_raw) begin : bad_data_accept_edge
              $fatal(1, "random stress load data mismatch txn %0d: exp=0x%0h got=0x%0h",
                     iter_var1, stress_data_raw, ld_data_data[ELEM_WIDTH-1:0]);
            end
            if (ld_done_data !== expected_tag) begin : bad_tag_accept_edge
              $fatal(1, "random stress lddone tag mismatch txn %0d: exp=%0d got=%0d",
                     iter_var1, expected_tag, ld_done_data);
            end
            seen_tag1 = 1'b1;
          end

          @(posedge clk);
          ld_addr_valid = 1'b0;

          if (ld_data_valid && ld_done_valid) begin : precheck_deassert_edge
            if (ld_data_data[ELEM_WIDTH-1:0] !== stress_data_raw) begin : bad_data_deassert_edge
              $fatal(1, "random stress load data mismatch txn %0d: exp=0x%0h got=0x%0h",
                     iter_var1, stress_data_raw, ld_data_data[ELEM_WIDTH-1:0]);
            end
            if (ld_done_data !== expected_tag) begin : bad_tag_deassert_edge
              $fatal(1, "random stress lddone tag mismatch txn %0d: exp=%0d got=%0d",
                     iter_var1, expected_tag, ld_done_data);
            end
            seen_tag1 = 1'b1;
          end

          iter_var0 = 0;
          while (iter_var0 < 40 && !seen_tag1) begin : wait_load_data
            @(posedge clk);
            if (ld_data_valid && ld_done_valid) begin : got_aligned
              if (ld_data_data[ELEM_WIDTH-1:0] !== stress_data_raw) begin : bad_data
                $fatal(1, "random stress load data mismatch txn %0d: exp=0x%0h got=0x%0h",
                       iter_var1, stress_data_raw, ld_data_data[ELEM_WIDTH-1:0]);
              end
              if (ld_done_data !== expected_tag) begin : bad_tag
                $fatal(1, "random stress lddone tag mismatch txn %0d: exp=%0d got=%0d",
                       iter_var1, expected_tag, ld_done_data);
              end
              seen_tag1 = 1'b1;
            end
            iter_var0 = iter_var0 + 1;
          end
          if (!seen_tag1) begin : check_load_data
            $fatal(1, "random stress load completion timeout (txn %0d)", iter_var1);
          end
        end else begin : native_load_roundtrip
          logic seen_ready;
          seen_ready = 1'b0;
          iter_var0 = 0;
          while (iter_var0 < 60 && !seen_tag1) begin : wait_load_data
            @(posedge clk);
            if (ld_addr_ready)
              seen_ready = 1'b1;
            if (ld_data_valid) begin : got_data
              if (^ld_data_data[ELEM_WIDTH-1:0] === 1'bx) begin : unknown_data
                // Ignore unknown transients and keep waiting for concrete data.
              end else if (ld_data_data[ELEM_WIDTH-1:0] !== stress_data_raw) begin : bad_data
                $fatal(1, "random stress load data mismatch txn %0d: exp=0x%0h got=0x%0h",
                       iter_var1, stress_data_raw, ld_data_data[ELEM_WIDTH-1:0]);
              end else begin : match_data
                seen_tag1 = 1'b1;
              end
            end
            iter_var0 = iter_var0 + 1;
          end
          ld_addr_valid = 1'b0;
          if (!seen_ready) begin : check_load_accept
            $fatal(1, "random stress load not accepted (txn %0d)", iter_var1);
          end
          if (!seen_tag1) begin : check_load_data
            $fatal(1, "random stress load completion timeout (txn %0d)", iter_var1);
          end
        end
      end

      pass_count = pass_count + 1;
    end

    // Check 11: OOB store raises error and does not modify memory state
    if (ST_COUNT >= 2 && TAG_WIDTH > 0 && LD_COUNT > 0) begin : oob_store_no_write_test
      rst_n = 0;
      ld_addr_valid = 1'b0;
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      ld_data_ready = 1'b1;
      ld_done_ready = 1'b1;
      st_done_ready = 1'b1;
      repeat (2) @(posedge clk);
      rst_n = 1;
      @(posedge clk);

      // Seed addr=50 with tag 0 store.
      st_addr_data = SAFE_ADDR_PW'(50);
      st_data_data = SAFE_ELEM_PW'(32'h5A5A);
      st_addr_valid = 1'b1;
      st_data_valid = 1'b1;
      iter_var0 = 0;
      while (iter_var0 < 10 && !(st_addr_ready && st_data_ready)) begin : wait_seed_ready_oob
        @(posedge clk);
        iter_var0 = iter_var0 + 1;
      end
      if (!(st_addr_ready && st_data_ready)) begin : check_seed_ready_oob
        $fatal(1, "seed store was not accepted in OOB store test");
      end
      @(posedge clk);
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      iter_var0 = 0;
      while (iter_var0 < 10) begin : wait_seed_done_oob
        @(posedge clk);
        iter_var0 = iter_var0 + 1;
        if (st_done_valid) begin : seed_done
          iter_var0 = 10;
        end
      end

      // Invalid tag = ST_COUNT: must report OOB and must not update state.
      st_addr_data = (SAFE_ADDR_PW'(ST_COUNT) << ADDR_WIDTH) | SAFE_ADDR_PW'(50);
      st_data_data = (SAFE_ELEM_PW'(ST_COUNT) << ELEM_WIDTH) | SAFE_ELEM_PW'(32'hDEAD);
      st_addr_valid = 1'b1;
      st_data_valid = 1'b1;
      @(posedge clk);
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      @(posedge clk);

      if (error_valid !== 1'b1 || error_code !== RT_MEMORY_TAG_OOB) begin : check_oob_store_error
        $fatal(1, "expected RT_MEMORY_TAG_OOB after OOB store, got valid=%0b code=%0d",
               error_valid, error_code);
      end

      ld_addr_data = SAFE_ADDR_PW'(50);
      ld_addr_valid = 1'b1;
      @(posedge clk);
      ld_addr_valid = 1'b0;
      if (ld_data_data[ELEM_WIDTH-1:0] !== ELEM_WIDTH'(32'h5A5A)) begin : check_preserved_data
        $fatal(1, "OOB store modified state: expected 0x5A5A, got 0x%0h",
               ld_data_data[ELEM_WIDTH-1:0]);
      end
      pass_count = pass_count + 1;
    end

    // Check 12: RT_MEMORY_STORE_DEADLOCK (only when ST_COUNT > 0)
    // Send only a store address (no data) and wait for deadlock timeout.
    if (ST_COUNT > 0) begin : deadlock_test
      rst_n = 0;
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      ld_addr_valid = 1'b0;
      repeat (2) @(posedge clk);
      rst_n = 1;
      @(posedge clk);
      // Enqueue only addr, leave data port idle -> triggers deadlock
      st_addr_data = SAFE_ADDR_PW'(1);
      st_addr_valid = 1'b1;
      @(posedge clk);
      st_addr_valid = 1'b0;
      // Wait for deadlock timeout + margin
      repeat (DEADLOCK_TIMEOUT + 4) @(posedge clk);
      if (error_valid !== 1'b1) begin : check_deadlock_valid
        $fatal(1, "expected RT_MEMORY_STORE_DEADLOCK error");
      end
      if (error_code !== RT_MEMORY_STORE_DEADLOCK) begin : check_deadlock_code
        $fatal(1, "wrong error code for deadlock: got %0d, expected %0d",
               error_code, RT_MEMORY_STORE_DEADLOCK);
      end
      pass_count = pass_count + 1;
    end

    // Check 13: exact deadlock boundary - error asserts exactly at TIMEOUT
    // Timeline: posedge N: addr enqueued. posedge N+1: counter starts (0->1).
    // posedge N+DT: counter reaches DT. posedge N+DT+1: error_latch captures.
    if (ST_COUNT > 0) begin : deadlock_exact_test
      rst_n = 0;
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      ld_addr_valid = 1'b0;
      st_done_ready = 1'b1;
      repeat (2) @(posedge clk);
      rst_n = 1;
      @(posedge clk);

      // Send only store address (no data) to create imbalance.
      st_addr_data = SAFE_ADDR_PW'(2);
      st_addr_valid = 1'b1;
      @(posedge clk);                        // posedge N: addr enqueued
      st_addr_valid = 1'b0;

      // Wait DT+1 posedges: DT for counter to reach threshold,
      // +1 for error_latch to capture on the next posedge.
      repeat (DEADLOCK_TIMEOUT + 1) @(posedge clk);
      @(negedge clk);
      #1;
      if (error_valid !== 1'b1) begin : exact_dl_valid
        $fatal(1, "exact deadlock: error_valid not set at expected time");
      end
      if (error_code !== RT_MEMORY_STORE_DEADLOCK) begin : exact_dl_code
        $fatal(1, "exact deadlock: wrong code %0d", error_code);
      end
      pass_count = pass_count + 1;
    end

    // Check 14: just below deadlock timeout - resolve before trigger
    if (ST_COUNT > 0) begin : deadlock_below_test
      rst_n = 0;
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      ld_addr_valid = 1'b0;
      st_done_ready = 1'b1;
      repeat (2) @(posedge clk);
      rst_n = 1;
      @(posedge clk);

      // Send only store address to create imbalance.
      st_addr_data = SAFE_ADDR_PW'(3);
      st_addr_valid = 1'b1;
      @(posedge clk);
      st_addr_valid = 1'b0;

      // Wait DEADLOCK_TIMEOUT - 2 cycles, then resolve by sending data.
      repeat (DEADLOCK_TIMEOUT - 2) @(posedge clk);

      st_data_data = SAFE_ELEM_PW'(32'h9999);
      st_data_valid = 1'b1;
      @(posedge clk);
      st_data_valid = 1'b0;

      // Wait a few more cycles and verify no error was raised.
      repeat (4) @(posedge clk);
      @(negedge clk);
      #1;
      if (error_valid !== 1'b0) begin : below_dl_check
        $fatal(1, "below timeout: error_valid should be 0 after resolving imbalance");
      end
      pass_count = pass_count + 1;
    end

    // Check 15: read-before-write same-cycle collision
    if (LD_COUNT > 0 && ST_COUNT > 0) begin : read_before_write_test
      rst_n = 0;
      ld_addr_valid = 1'b0;
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      ld_data_ready = 1'b1;
      ld_done_ready = 1'b1;
      st_done_ready = 1'b1;
      repeat (2) @(posedge clk);
      rst_n = 1;
      @(posedge clk);

      // Store value 0xAAAA to address 7.
      st_addr_data = SAFE_ADDR_PW'(7);
      st_data_data = SAFE_ELEM_PW'(32'hAAAA);
      st_addr_valid = 1'b1;
      st_data_valid = 1'b1;
      iter_var0 = 0;
      while (iter_var0 < 20) begin : wait_rbw_stdone
        @(posedge clk);
        iter_var0 = iter_var0 + 1;
        if (st_done_valid) begin : done
          iter_var0 = 20;
        end
      end
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;
      @(posedge clk);

      // Now in the same cycle: issue load to addr 7 AND start a store of 0xBBBB to addr 7.
      // Load path is combinational (reads old mem value), store writes on posedge.
      ld_addr_data = SAFE_ADDR_PW'(7);
      ld_addr_valid = 1'b1;
      st_addr_data = SAFE_ADDR_PW'(7);
      st_data_data = SAFE_ELEM_PW'(32'hBBBB);
      st_addr_valid = 1'b1;
      st_data_valid = 1'b1;

      // Check load returns old value (read-before-write).
      #1;
      if (ld_data_valid) begin : check_rbw_old
        if (ld_data_data[ELEM_WIDTH-1:0] !== ELEM_WIDTH'(32'hAAAA)) begin : bad_rbw
          $fatal(1, "read-before-write: load should return old value 0xAAAA, got 0x%0h",
                 ld_data_data[ELEM_WIDTH-1:0]);
        end
      end
      @(posedge clk);
      ld_addr_valid = 1'b0;
      st_addr_valid = 1'b0;
      st_data_valid = 1'b0;

      // Wait for store to complete.
      iter_var0 = 0;
      while (iter_var0 < 20) begin : wait_rbw_stdone2
        @(posedge clk);
        iter_var0 = iter_var0 + 1;
        if (st_done_valid) begin : done
          iter_var0 = 20;
        end
      end

      // Read addr 7 again: should be new value 0xBBBB.
      ld_addr_data = SAFE_ADDR_PW'(7);
      ld_addr_valid = 1'b1;
      @(posedge clk);
      ld_addr_valid = 1'b0;
      if (ld_data_data[ELEM_WIDTH-1:0] !== ELEM_WIDTH'(32'hBBBB)) begin : check_rbw_new
        $fatal(1, "read-before-write: second load should return 0xBBBB, got 0x%0h",
               ld_data_data[ELEM_WIDTH-1:0]);
      end
      pass_count = pass_count + 1;
    end

    $display("PASS: tb_fabric_memory AW=%0d EW=%0d TW=%0d LD=%0d ST=%0d (%0d checks)",
             ADDR_WIDTH, ELEM_WIDTH, TAG_WIDTH, LD_COUNT, ST_COUNT, pass_count);
    $finish;
  end

  initial begin : timeout
    // Allow enough time for 3 deadlock-related tests + other checks
    #((4 * DEADLOCK_TIMEOUT + 2000) * 10);
    $fatal(1, "TIMEOUT");
  end
endmodule
