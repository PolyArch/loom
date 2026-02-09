//===-- tb_fabric_extmemory.sv - Parameterized extmemory test ---*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_fabric_extmemory;
  parameter int DATA_WIDTH       = 32;
  parameter int TAG_WIDTH        = 0;
  parameter int LD_COUNT         = 1;
  parameter int ST_COUNT         = 1;
  parameter int LSQ_DEPTH        = 4;
  parameter int DEADLOCK_TIMEOUT = 65535;

  localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH;
  localparam int SAFE_PW = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1;
  localparam int SAFE_TW = (TAG_WIDTH > 0) ? TAG_WIDTH : 1;
  // Extmemory: first input is memref binding
  localparam int NUM_INPUTS  = 1 + LD_COUNT + 2 * ST_COUNT;
  localparam int NUM_OUTPUTS = LD_COUNT + 1 + (ST_COUNT > 0 ? 1 : 0);

  logic clk, rst_n;
  logic [NUM_INPUTS-1:0]                in_valid;
  logic [NUM_INPUTS-1:0]                in_ready;
  logic [NUM_INPUTS-1:0][SAFE_PW-1:0]  in_data;
  logic [NUM_OUTPUTS-1:0]              out_valid;
  logic [NUM_OUTPUTS-1:0]              out_ready;
  logic [NUM_OUTPUTS-1:0][SAFE_PW-1:0] out_data;
  logic        error_valid;
  logic [15:0] error_code;

  fabric_extmemory #(
    .DATA_WIDTH(DATA_WIDTH),
    .TAG_WIDTH(TAG_WIDTH),
    .LD_COUNT(LD_COUNT),
    .ST_COUNT(ST_COUNT),
    .LSQ_DEPTH(LSQ_DEPTH),
    .DEADLOCK_TIMEOUT(DEADLOCK_TIMEOUT)
  ) dut (
    .clk(clk), .rst_n(rst_n),
    .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data),
    .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
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
    in_valid = '0;
    out_ready = '1;
    in_data = '0;

    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // Check 1: no error after reset
    if (error_valid !== 1'b0) begin : check_reset
      $fatal(1, "error_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    // Check 2: store then load - verify round-trip through FIFO-based store queue
    // Port layout (inputs): [memref(0)] [ld_addr(1..LD)] [st_addr] [st_data]
    if (ST_COUNT > 0 && LD_COUNT > 0) begin : store_load
      in_data = '0;
      // st_addr port index = 1 + LD_COUNT + 0
      in_data[1 + LD_COUNT][SAFE_PW-1:0] = SAFE_PW'(5);
      // st_data port index = 1 + LD_COUNT + ST_COUNT + 0
      in_data[1 + LD_COUNT + ST_COUNT][SAFE_PW-1:0] = SAFE_PW'(32'hABCD);
      in_valid = '0;
      in_valid[1 + LD_COUNT] = 1'b1;
      in_valid[1 + LD_COUNT + ST_COUNT] = 1'b1;
      @(posedge clk);
      in_valid = '0;
      // Wait for store done (stdone at output index LD_COUNT + 1)
      iter_var0 = 0;
      while (iter_var0 < 10) begin : wait_stdone
        @(posedge clk);
        iter_var0 = iter_var0 + 1;
        if (out_valid[LD_COUNT + 1]) begin : done
          iter_var0 = 10;
        end
      end
      @(posedge clk);

      // Load: ld_addr port index = 1 (skip memref)
      in_data = '0;
      in_data[1][SAFE_PW-1:0] = SAFE_PW'(5);
      in_valid = '0;
      in_valid[1] = 1'b1;
      @(posedge clk);
      if (out_valid[0] !== 1'b1) begin : check_ld_valid
        $fatal(1, "load data output should be valid");
      end
      if (out_data[0][DATA_WIDTH-1:0] !== DATA_WIDTH'(32'hABCD)) begin : check_ld_data
        $fatal(1, "load data mismatch: expected 0xABCD, got 0x%0h",
               out_data[0][DATA_WIDTH-1:0]);
      end
      pass_count = pass_count + 1;
      in_valid = '0;
      @(posedge clk);
    end

    // Check 3: no error after normal operation
    if (error_valid !== 1'b0) begin : check_no_err
      $fatal(1, "unexpected error after store/load: code=%0d", error_code);
    end
    pass_count = pass_count + 1;

    // Check 4: multi-port store via different tags (when ST_COUNT >= 2 && TAG_WIDTH > 0)
    // Store on port 0 (tag=0) and port 1 (tag=1) sequentially, then verify.
    if (ST_COUNT >= 2 && TAG_WIDTH > 0) begin : cross_port_test
      rst_n = 0;
      in_valid = '0;
      repeat (2) @(posedge clk);
      rst_n = 1;
      @(posedge clk);

      // Store via port 0 (tag=0): addr=10, data=0xDEAD
      in_data = '0;
      in_data[1 + LD_COUNT] = SAFE_PW'(10);
      in_data[1 + LD_COUNT + ST_COUNT] = SAFE_PW'(32'hDEAD);
      in_valid = '0;
      in_valid[1 + LD_COUNT] = 1'b1;
      in_valid[1 + LD_COUNT + ST_COUNT] = 1'b1;
      @(posedge clk);
      in_valid = '0;
      repeat (10) @(posedge clk);

      // Store via port 1 (tag=1): addr=11, data=0xBEEF
      in_data = '0;
      in_data[1 + LD_COUNT + 1] = (SAFE_PW'(1) << DATA_WIDTH) | SAFE_PW'(11);
      in_data[1 + LD_COUNT + ST_COUNT + 1] = (SAFE_PW'(1) << DATA_WIDTH) | SAFE_PW'(32'hBEEF);
      in_valid = '0;
      in_valid[1 + LD_COUNT + 1] = 1'b1;
      in_valid[1 + LD_COUNT + ST_COUNT + 1] = 1'b1;
      @(posedge clk);
      in_valid = '0;
      repeat (10) @(posedge clk);

      // Load addr=10 -> expect 0xDEAD
      in_data = '0;
      in_data[1][SAFE_PW-1:0] = SAFE_PW'(10);
      in_valid = '0;
      in_valid[1] = 1'b1;
      @(posedge clk);
      in_valid = '0;
      if (out_data[0][DATA_WIDTH-1:0] !== DATA_WIDTH'(32'hDEAD)) begin : check_st0
        $fatal(1, "cross-port store 0: expected 0xDEAD, got 0x%0h",
               out_data[0][DATA_WIDTH-1:0]);
      end
      @(posedge clk);

      // Load addr=11 -> expect 0xBEEF
      in_data[1][SAFE_PW-1:0] = SAFE_PW'(11);
      in_valid[1] = 1'b1;
      @(posedge clk);
      in_valid = '0;
      if (out_data[0][DATA_WIDTH-1:0] !== DATA_WIDTH'(32'hBEEF)) begin : check_st1
        $fatal(1, "cross-port store 1: expected 0xBEEF, got 0x%0h",
               out_data[0][DATA_WIDTH-1:0]);
      end
      pass_count = pass_count + 1;
    end

    // Check 5: stdone tag correctness (when ST_COUNT >= 2 && TAG_WIDTH > 0)
    // Verify stdone output carries the correct tag index.
    if (ST_COUNT >= 2 && TAG_WIDTH > 0) begin : stdone_tag_test
      rst_n = 0;
      in_valid = '0;
      repeat (2) @(posedge clk);
      rst_n = 1;
      @(posedge clk);

      // Store on port 1 targeting tag 1: addr=20, data=0x1234
      in_data = '0;
      // Tag goes in the upper TAG_WIDTH bits of the payload
      in_data[1 + LD_COUNT + 1] = (SAFE_PW'(1) << DATA_WIDTH) | SAFE_PW'(20);
      in_data[1 + LD_COUNT + ST_COUNT + 1] = (SAFE_PW'(1) << DATA_WIDTH) | SAFE_PW'(32'h1234);
      in_valid = '0;
      in_valid[1 + LD_COUNT + 1] = 1'b1;
      in_valid[1 + LD_COUNT + ST_COUNT + 1] = 1'b1;
      @(posedge clk);
      in_valid = '0;

      // Wait for stdone and check tag
      iter_var0 = 0;
      while (iter_var0 < 10) begin : wait_stdone_tag
        @(posedge clk);
        iter_var0 = iter_var0 + 1;
        if (out_valid[LD_COUNT + 1]) begin : got_stdone
          // stdone tag should be 1 (the tag index of the paired store)
          if (out_data[LD_COUNT + 1][SAFE_PW-1 -: SAFE_TW] !== SAFE_TW'(1)) begin : bad_tag
            $fatal(1, "stdone tag: expected 1, got %0d",
                   out_data[LD_COUNT + 1][SAFE_PW-1 -: SAFE_TW]);
          end
          iter_var0 = 10;
        end
      end
      pass_count = pass_count + 1;
    end

    // Check 6: cross-port same-tag contention (when ST_COUNT >= 2 && TAG_WIDTH > 0)
    // Both port 0 and port 1 simultaneously target tag=0. Per-tag grant allows
    // only one per cycle; the denied port retries and succeeds next cycle.
    if (ST_COUNT >= 2 && TAG_WIDTH > 0) begin : same_tag_test
      rst_n = 0;
      in_valid = '0;
      repeat (2) @(posedge clk);
      rst_n = 1;
      @(posedge clk);

      // Simultaneously present tag=0 addr+data on port 0 and port 1.
      // Port 0: addr=30, data=0xAAAA (tag=0)
      // Port 1: addr=31, data=0xBBBB (tag=0)
      in_data = '0;
      in_data[1 + LD_COUNT]              = SAFE_PW'(30);       // port 0 addr, tag=0
      in_data[1 + LD_COUNT + ST_COUNT]   = SAFE_PW'(32'hAAAA); // port 0 data, tag=0
      in_data[1 + LD_COUNT + 1]          = SAFE_PW'(31);       // port 1 addr, tag=0
      in_data[1 + LD_COUNT + ST_COUNT+1] = SAFE_PW'(32'hBBBB); // port 1 data, tag=0
      in_valid = '0;
      in_valid[1 + LD_COUNT]            = 1'b1;
      in_valid[1 + LD_COUNT + ST_COUNT] = 1'b1;
      in_valid[1 + LD_COUNT + 1]        = 1'b1;
      in_valid[1 + LD_COUNT + ST_COUNT+1] = 1'b1;
      @(posedge clk);
      // Port 0 granted, port 1 denied. Deassert port 0, keep port 1 for retry.
      in_data[1 + LD_COUNT]            = '0;
      in_data[1 + LD_COUNT + ST_COUNT] = '0;
      in_valid[1 + LD_COUNT]           = 1'b0;
      in_valid[1 + LD_COUNT + ST_COUNT] = 1'b0;
      @(posedge clk);
      in_valid = '0;
      repeat (10) @(posedge clk);

      // Load addr=30 -> expect 0xAAAA
      in_data = '0;
      in_data[1][SAFE_PW-1:0] = SAFE_PW'(30);
      in_valid = '0;
      in_valid[1] = 1'b1;
      @(posedge clk);
      in_valid = '0;
      if (out_data[0][DATA_WIDTH-1:0] !== DATA_WIDTH'(32'hAAAA)) begin : check_st0_same
        $fatal(1, "same-tag store 0: expected 0xAAAA, got 0x%0h",
               out_data[0][DATA_WIDTH-1:0]);
      end
      @(posedge clk);

      // Load addr=31 -> expect 0xBBBB
      in_data[1][SAFE_PW-1:0] = SAFE_PW'(31);
      in_valid[1] = 1'b1;
      @(posedge clk);
      in_valid = '0;
      if (out_data[0][DATA_WIDTH-1:0] !== DATA_WIDTH'(32'hBBBB)) begin : check_st1_same
        $fatal(1, "same-tag store 1: expected 0xBBBB, got 0x%0h",
               out_data[0][DATA_WIDTH-1:0]);
      end
      pass_count = pass_count + 1;
    end

    // Check 7: deadlock detection (when ST_COUNT > 0)
    if (ST_COUNT > 0) begin : deadlock_test
      rst_n = 0;
      in_valid = '0;
      repeat (2) @(posedge clk);
      rst_n = 1;
      @(posedge clk);
      // Enqueue only addr, leave data FIFO empty -> triggers deadlock
      in_data = '0;
      in_data[1 + LD_COUNT][SAFE_PW-1:0] = SAFE_PW'(1);
      in_valid = '0;
      in_valid[1 + LD_COUNT] = 1'b1;
      @(posedge clk);
      in_valid = '0;
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

    $display("PASS: tb_fabric_extmemory DW=%0d TW=%0d LD=%0d ST=%0d (%0d checks)",
             DATA_WIDTH, TAG_WIDTH, LD_COUNT, ST_COUNT, pass_count);
    $finish;
  end

  initial begin : timeout
    #((DEADLOCK_TIMEOUT + 250) * 10);
    $fatal(1, "TIMEOUT");
  end
endmodule
