//===-- tb_fabric_memory.sv - Parameterized memory test --------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_fabric_memory;
  parameter int DATA_WIDTH       = 32;
  parameter int TAG_WIDTH        = 0;
  parameter int LD_COUNT         = 1;
  parameter int ST_COUNT         = 1;
  parameter int LSQ_DEPTH        = 4;
  parameter int IS_PRIVATE       = 1;
  parameter int MEM_DEPTH        = 64;
  parameter int DEADLOCK_TIMEOUT = 65535;

  localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH;
  localparam int SAFE_PW = (PAYLOAD_WIDTH > 0) ? PAYLOAD_WIDTH : 1;
  localparam int NUM_INPUTS  = LD_COUNT + 2 * ST_COUNT;
  localparam int NUM_OUTPUTS = (IS_PRIVATE ? 0 : 1) + LD_COUNT + 1 + (ST_COUNT > 0 ? 1 : 0);

  logic clk, rst_n;
  logic [NUM_INPUTS-1:0]                in_valid;
  logic [NUM_INPUTS-1:0]                in_ready;
  logic [NUM_INPUTS-1:0][SAFE_PW-1:0]  in_data;
  logic [NUM_OUTPUTS-1:0]              out_valid;
  logic [NUM_OUTPUTS-1:0]              out_ready;
  logic [NUM_OUTPUTS-1:0][SAFE_PW-1:0] out_data;
  logic        error_valid;
  logic [15:0] error_code;

  fabric_memory #(
    .DATA_WIDTH(DATA_WIDTH),
    .TAG_WIDTH(TAG_WIDTH),
    .LD_COUNT(LD_COUNT),
    .ST_COUNT(ST_COUNT),
    .LSQ_DEPTH(LSQ_DEPTH),
    .IS_PRIVATE(IS_PRIVATE),
    .MEM_DEPTH(MEM_DEPTH),
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
    if (ST_COUNT > 0 && LD_COUNT > 0) begin : store_load
      // Store: addr=5, data=0xABCD
      // Input layout: [ld_addr * LD_COUNT, st_addr * ST_COUNT, st_data * ST_COUNT]
      // With FIFO-based pairing: addr and data are enqueued independently,
      // then paired internally and written to memory.
      in_data = '0;
      in_data[LD_COUNT][SAFE_PW-1:0] = SAFE_PW'(5);
      in_data[LD_COUNT + ST_COUNT][SAFE_PW-1:0] = SAFE_PW'(32'hABCD);
      in_valid = '0;
      in_valid[LD_COUNT] = 1'b1;
      in_valid[LD_COUNT + ST_COUNT] = 1'b1;
      @(posedge clk);
      // Addr and data are accepted into FIFOs immediately
      in_valid = '0;
      // Wait for store done (pair_fire -> write_mem -> stdone)
      iter_var0 = 0;
      while (iter_var0 < 10) begin : wait_stdone
        @(posedge clk);
        iter_var0 = iter_var0 + 1;
        if (out_valid[(IS_PRIVATE ? 0 : 1) + LD_COUNT + 1]) begin : done
          iter_var0 = 10;
        end
      end
      @(posedge clk);

      // Load: addr=5
      in_data = '0;
      in_data[0][SAFE_PW-1:0] = SAFE_PW'(5);
      in_valid = '0;
      in_valid[0] = 1'b1;
      @(posedge clk);
      if (out_valid[IS_PRIVATE ? 0 : 1] !== 1'b1) begin : check_ld_valid
        $fatal(1, "load data output should be valid");
      end
      if (out_data[IS_PRIVATE ? 0 : 1][DATA_WIDTH-1:0] !== DATA_WIDTH'(32'hABCD)) begin : check_ld_data
        $fatal(1, "load data mismatch: expected 0xABCD, got 0x%0h",
               out_data[IS_PRIVATE ? 0 : 1][DATA_WIDTH-1:0]);
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

    // Check 4: RT_MEMORY_TAG_OOB (only testable when TAG_WIDTH > 0 and LD_COUNT > 1)
    // Uses SAFE_TW to avoid zero-width part-select when TAG_WIDTH=0.
    if (TAG_WIDTH > 0 && LD_COUNT > 1) begin : tag_oob_test
      rst_n = 0;
      repeat (2) @(posedge clk);
      rst_n = 1;
      @(posedge clk);
      // Send load with tag = LD_COUNT (out of bounds)
      // Use SAFE_PW-wide assignment to avoid zero-width TAG_WIDTH part-select
      in_data = '0;
      in_data[0] = SAFE_PW'(LD_COUNT) << DATA_WIDTH;
      in_valid = '0;
      in_valid[0] = 1'b1;
      @(posedge clk);
      @(posedge clk);
      if (error_valid !== 1'b1) begin : check_tag_oob
        $fatal(1, "expected RT_MEMORY_TAG_OOB error");
      end
      if (error_code !== RT_MEMORY_TAG_OOB) begin : check_tag_oob_code
        $fatal(1, "wrong error code for tag OOB: got %0d", error_code);
      end
      pass_count = pass_count + 1;
      in_valid = '0;
    end

    // Check 5: RT_MEMORY_STORE_DEADLOCK (only when ST_COUNT > 0)
    // Send only a store address (no data) and wait for deadlock timeout.
    if (ST_COUNT > 0) begin : deadlock_test
      rst_n = 0;
      in_valid = '0;
      repeat (2) @(posedge clk);
      rst_n = 1;
      @(posedge clk);
      // Enqueue only addr, leave data FIFO empty -> triggers deadlock
      in_data = '0;
      in_data[LD_COUNT][SAFE_PW-1:0] = SAFE_PW'(1);
      in_valid = '0;
      in_valid[LD_COUNT] = 1'b1;
      @(posedge clk);
      in_valid = '0;
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

    $display("PASS: tb_fabric_memory DW=%0d TW=%0d LD=%0d ST=%0d (%0d checks)",
             DATA_WIDTH, TAG_WIDTH, LD_COUNT, ST_COUNT, pass_count);
    $finish;
  end

  initial begin : timeout
    // Allow enough time for deadlock test (DEADLOCK_TIMEOUT + margin cycles)
    #((DEADLOCK_TIMEOUT + 200) * 10);
    $fatal(1, "TIMEOUT");
  end
endmodule
