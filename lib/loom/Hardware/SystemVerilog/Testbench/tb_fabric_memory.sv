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

    // Check 8: RT_MEMORY_STORE_DEADLOCK (only when ST_COUNT > 0)
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

    $display("PASS: tb_fabric_memory AW=%0d EW=%0d TW=%0d LD=%0d ST=%0d (%0d checks)",
             ADDR_WIDTH, ELEM_WIDTH, TAG_WIDTH, LD_COUNT, ST_COUNT, pass_count);
    $finish;
  end

  initial begin : timeout
    // Allow enough time for deadlock test (DEADLOCK_TIMEOUT + margin cycles)
    #((DEADLOCK_TIMEOUT + 200) * 10);
    $fatal(1, "TIMEOUT");
  end
endmodule
