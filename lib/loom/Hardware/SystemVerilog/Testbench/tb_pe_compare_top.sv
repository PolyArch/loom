//===-- tb_pe_compare_top.sv - E2E test for runtime-configurable cmp -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Testbench for pe_compare_top, verifying:
//   1. All 10 cmpi predicates (0-9) on p0
//   2. Key cmpf predicates on p1 (including NaN handling)
//   3. Invalid cmpi predicate error detection (predicate >= 10)
//   4. Multi-cmp body (p2) with independent cmpi/cmpf predicate control
//   5. Runtime predicate switching
//
//===----------------------------------------------------------------------===//

`include "fabric_error.svh"

module tb_pe_compare_top;

  logic        clk;
  logic        rst_n;

  // p0 ports: arith.cmpi (i32, i32) -> i1
  logic        a0_valid, a0_ready;
  logic [31:0] a0_data;
  logic        b0_valid, b0_ready;
  logic [31:0] b0_data;
  logic        r0_valid, r0_ready;
  logic        r0_data;

  // p1 ports: arith.cmpf (f32, f32) -> i1
  logic        a1_valid, a1_ready;
  logic [31:0] a1_data;
  logic        b1_valid, b1_ready;
  logic [31:0] b1_data;
  logic        r1_valid, r1_ready;
  logic        r1_data;

  // p2 ports: multi-cmp (i32, i32, f32, f32) -> (i1, i1)
  logic        a2_valid, a2_ready;
  logic [31:0] a2_data;
  logic        b2_valid, b2_ready;
  logic [31:0] b2_data;
  logic        c2_valid, c2_ready;
  logic [31:0] c2_data;
  logic        d2_valid, d2_ready;
  logic [31:0] d2_data;
  logic        r2_i_valid, r2_i_ready;
  logic        r2_i_data;
  logic        r2_f_valid, r2_f_ready;
  logic        r2_f_data;

  // Config: p0 has 4-bit cfg (cmpi pred), p1 has 4-bit cfg (cmpf pred),
  // p2 has 8-bit cfg (cmpi[3:0] + cmpf[7:4])
  logic [3:0]  p0_cfg_data;
  logic [3:0]  p1_cfg_data;
  logic [7:0]  p2_cfg_data;

  // Error ports (aggregate from p0 and p2 which have cmpi)
  logic        error_valid;
  logic [15:0] error_code;

  pe_compare_top dut (
    .clk         (clk),
    .rst_n       (rst_n),
    .a0_valid    (a0_valid),
    .a0_ready    (a0_ready),
    .a0_data     (a0_data),
    .b0_valid    (b0_valid),
    .b0_ready    (b0_ready),
    .b0_data     (b0_data),
    .r0_valid    (r0_valid),
    .r0_ready    (r0_ready),
    .r0_data     (r0_data),
    .a1_valid    (a1_valid),
    .a1_ready    (a1_ready),
    .a1_data     (a1_data),
    .b1_valid    (b1_valid),
    .b1_ready    (b1_ready),
    .b1_data     (b1_data),
    .r1_valid    (r1_valid),
    .r1_ready    (r1_ready),
    .r1_data     (r1_data),
    .a2_valid    (a2_valid),
    .a2_ready    (a2_ready),
    .a2_data     (a2_data),
    .b2_valid    (b2_valid),
    .b2_ready    (b2_ready),
    .b2_data     (b2_data),
    .c2_valid    (c2_valid),
    .c2_ready    (c2_ready),
    .c2_data     (c2_data),
    .d2_valid    (d2_valid),
    .d2_ready    (d2_ready),
    .d2_data     (d2_data),
    .r2_i_valid  (r2_i_valid),
    .r2_i_ready  (r2_i_ready),
    .r2_i_data   (r2_i_data),
    .r2_f_valid  (r2_f_valid),
    .r2_f_ready  (r2_f_ready),
    .r2_f_data   (r2_f_data),
    .p0_cfg_data (p0_cfg_data),
    .p1_cfg_data (p1_cfg_data),
    .p2_cfg_data (p2_cfg_data),
    .error_valid (error_valid),
    .error_code  (error_code)
  );

  initial begin : clk_gen
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_pe_compare_top);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_pe_compare_top, "+mda");
  end
`endif

  // IEEE 754 constants
  localparam logic [31:0] F32_1_5 = 32'h3FC00000; // 1.5f
  localparam logic [31:0] F32_2_0 = 32'h40000000; // 2.0f
  localparam logic [31:0] F32_3_0 = 32'h40400000; // 3.0f
  localparam logic [31:0] F32_5_0 = 32'h40A00000; // 5.0f
  localparam logic [31:0] F32_NAN = 32'h7FC00000; // quiet NaN

  // Drive p0 (cmpi) inputs and wait for result
  task automatic drive_cmpi(
      input logic [31:0] a, input logic [31:0] b,
      input logic expected);
    integer iter_var0;
    logic seen;
    begin : drive_cmpi_task
      @(negedge clk);
      a0_data  = a;
      b0_data  = b;
      a0_valid = 1'b1;
      b0_valid = 1'b1;

      seen = 1'b0;
      iter_var0 = 0;
      while (iter_var0 < 40 && !seen) begin : wait_r0
        @(posedge clk);
        #1;
        if (r0_valid) begin : check_r0
          if (r0_data !== expected) begin : bad_r0
            $fatal(1, "cmpi mismatch: pred=%0d a=%0d b=%0d expected=%0b got=%0b",
                   p0_cfg_data, $signed(a), $signed(b), expected, r0_data);
          end
          seen = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!seen) begin : timeout_r0
        $fatal(1, "cmpi timeout: pred=%0d a=%0d b=%0d", p0_cfg_data, $signed(a), $signed(b));
      end

      @(negedge clk);
      a0_valid = 1'b0;
      b0_valid = 1'b0;
    end
  endtask

  // Drive p1 (cmpf) inputs and wait for result
  task automatic drive_cmpf(
      input logic [31:0] a, input logic [31:0] b,
      input logic expected);
    integer iter_var0;
    logic seen;
    begin : drive_cmpf_task
      @(negedge clk);
      a1_data  = a;
      b1_data  = b;
      a1_valid = 1'b1;
      b1_valid = 1'b1;

      seen = 1'b0;
      iter_var0 = 0;
      while (iter_var0 < 40 && !seen) begin : wait_r1
        @(posedge clk);
        #1;
        if (r1_valid) begin : check_r1
          if (r1_data !== expected) begin : bad_r1
            $fatal(1, "cmpf mismatch: pred=%0d a=0x%08h b=0x%08h expected=%0b got=%0b",
                   p1_cfg_data, a, b, expected, r1_data);
          end
          seen = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!seen) begin : timeout_r1
        $fatal(1, "cmpf timeout: pred=%0d a=0x%08h b=0x%08h", p1_cfg_data, a, b);
      end

      @(negedge clk);
      a1_valid = 1'b0;
      b1_valid = 1'b0;
    end
  endtask

  // Drive p2 (multi-cmp) inputs and wait for both results
  // a,b are i32 for cmpi; c,d are f32 (bit pattern) for cmpf
  task automatic drive_multi_cmp(
      input logic [31:0] a, input logic [31:0] b,
      input logic [31:0] c, input logic [31:0] d,
      input logic expected_i, input logic expected_f);
    integer iter_var0;
    logic seen;
    begin : drive_multi_task
      @(negedge clk);
      a2_data  = a;
      b2_data  = b;
      c2_data  = c;
      d2_data  = d;
      a2_valid = 1'b1;
      b2_valid = 1'b1;
      c2_valid = 1'b1;
      d2_valid = 1'b1;

      seen = 1'b0;
      iter_var0 = 0;
      while (iter_var0 < 40 && !seen) begin : wait_r2
        @(posedge clk);
        #1;
        if (r2_i_valid && r2_f_valid) begin : check_r2
          if (r2_i_data !== expected_i) begin : bad_r2_i
            $fatal(1, "multi cmpi mismatch: cfg=0x%02h a=%0d b=%0d expected=%0b got=%0b",
                   p2_cfg_data, $signed(a), $signed(b), expected_i, r2_i_data);
          end
          if (r2_f_data !== expected_f) begin : bad_r2_f
            $fatal(1, "multi cmpf mismatch: cfg=0x%02h c=0x%08h d=0x%08h expected=%0b got=%0b",
                   p2_cfg_data, c, d, expected_f, r2_f_data);
          end
          seen = 1'b1;
        end
        iter_var0 = iter_var0 + 1;
      end
      if (!seen) begin : timeout_r2
        $fatal(1, "multi cmp timeout: cfg=0x%02h a=%0d b=%0d", p2_cfg_data, $signed(a), $signed(b));
      end

      @(negedge clk);
      a2_valid = 1'b0;
      b2_valid = 1'b0;
      c2_valid = 1'b0;
      d2_valid = 1'b0;
    end
  endtask

  initial begin : main
    integer pass_count;
    integer iter_var0;
    logic [9:0] cmpi_expect;
    logic [15:0] cmpf_expect;
    logic [15:0] cmpf_nan_expect;

    // cmpi expected results for (a=5, b=3), predicates 0-9:
    //   eq=0, ne=1, slt=0, sle=0, sgt=1, sge=1, ult=0, ule=0, ugt=1, uge=1
    cmpi_expect = {1'b1, 1'b1, 1'b0, 1'b0, 1'b1, 1'b1, 1'b0, 1'b0, 1'b1, 1'b0};

    // cmpf expected results for (1.5, 2.0), predicates 0-15:
    //   false=0, oeq=0, ogt=0, oge=0, olt=1, ole=1, one=1, ord=1,
    //   ueq=0, ugt=0, uge=0, ult=1, ule=1, une=1, uno=0, true=1
    cmpf_expect = {1'b1, 1'b0, 1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 1'b0,
                   1'b1, 1'b1, 1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 1'b0};

    // cmpf expected results for (NaN, 2.0), predicates 0-15:
    //   false=0, oeq=0, ogt=0, oge=0, olt=0, ole=0, one=0, ord=0,
    //   ueq=1, ugt=1, uge=1, ult=1, ule=1, une=1, uno=1, true=1
    cmpf_nan_expect = {1'b1, 1'b1, 1'b1, 1'b1, 1'b1, 1'b1, 1'b1, 1'b1,
                       1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0};

    pass_count = 0;

    rst_n = 1'b0;
    a0_valid = 1'b0; a0_data = '0;
    b0_valid = 1'b0; b0_data = '0;
    a1_valid = 1'b0; a1_data = '0;
    b1_valid = 1'b0; b1_data = '0;
    a2_valid = 1'b0; a2_data = '0;
    b2_valid = 1'b0; b2_data = '0;
    c2_valid = 1'b0; c2_data = '0;
    d2_valid = 1'b0; d2_data = '0;
    r0_ready = 1'b1;
    r1_ready = 1'b1;
    r2_i_ready = 1'b1;
    r2_f_ready = 1'b1;
    p0_cfg_data = 4'd2;  // slt
    p1_cfg_data = 4'd4;  // olt
    p2_cfg_data = 8'h42; // cmpi=slt(2), cmpf=olt(4)

    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // ---------------------------------------------------------------
    // Test 1: All 10 cmpi predicates with a=5, b=3
    // ---------------------------------------------------------------
    $display("Test 1: cmpi predicates 0-9");
    iter_var0 = 0;
    while (iter_var0 < 10) begin : cmpi_loop
      @(negedge clk);
      p0_cfg_data = iter_var0[3:0];
      @(posedge clk);
      drive_cmpi(32'd5, 32'd3, cmpi_expect[iter_var0]);
      iter_var0 = iter_var0 + 1;
    end
    pass_count = pass_count + 1;

    // ---------------------------------------------------------------
    // Test 2: All 16 cmpf predicates with 1.5 vs 2.0
    // ---------------------------------------------------------------
    $display("Test 2: cmpf predicates 0-15 (1.5 vs 2.0)");
    iter_var0 = 0;
    while (iter_var0 < 16) begin : cmpf_loop
      @(negedge clk);
      p1_cfg_data = iter_var0[3:0];
      @(posedge clk);
      drive_cmpf(F32_1_5, F32_2_0, cmpf_expect[iter_var0]);
      iter_var0 = iter_var0 + 1;
    end
    pass_count = pass_count + 1;

    // ---------------------------------------------------------------
    // Test 3: NaN handling for cmpf (NaN vs 2.0)
    // ---------------------------------------------------------------
    $display("Test 3: cmpf NaN handling");
    iter_var0 = 0;
    while (iter_var0 < 16) begin : cmpf_nan_loop
      @(negedge clk);
      p1_cfg_data = iter_var0[3:0];
      @(posedge clk);
      drive_cmpf(F32_NAN, F32_2_0, cmpf_nan_expect[iter_var0]);
      iter_var0 = iter_var0 + 1;
    end
    pass_count = pass_count + 1;

    // ---------------------------------------------------------------
    // Test 4: Invalid cmpi predicate error detection (10-15)
    //   Error latch captures first error and holds until reset.
    // ---------------------------------------------------------------
    $display("Test 4: cmpi invalid predicate error");

    // 4a: predicate=10 triggers error
    @(negedge clk);
    p0_cfg_data = 4'd10;
    @(posedge clk);
    #1;
    if (!error_valid) begin : no_err_10
      $fatal(1, "error_valid not asserted for cmpi predicate=10");
    end
    if (error_code !== CFG_PE_CMPI_PREDICATE_INVALID) begin : wrong_code_10
      $fatal(1, "wrong error_code: expected %0d got %0d",
             CFG_PE_CMPI_PREDICATE_INVALID, error_code);
    end

    // 4b: reset clears the error latch
    @(negedge clk);
    rst_n = 1'b0;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // 4c: predicate=15 also triggers error
    @(negedge clk);
    p0_cfg_data = 4'd15;
    @(posedge clk);
    #1;
    if (!error_valid) begin : no_err_15
      $fatal(1, "error_valid not asserted for cmpi predicate=15");
    end

    // 4d: reset and restore valid predicate; error should stay clear
    @(negedge clk);
    rst_n = 1'b0;
    p0_cfg_data = 4'd0;
    p2_cfg_data = 8'h00;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
    #1;
    if (error_valid) begin : err_stuck
      $fatal(1, "error_valid asserted with valid predicate after reset");
    end
    pass_count = pass_count + 1;

    // ---------------------------------------------------------------
    // Test 5: Multi-cmp body (p2) with independent predicate control
    //   cfg_data layout: [7:4]=cmpf_pred, [3:0]=cmpi_pred
    //   a=5, b=3 as i32 for cmpi; c=5.0f, d=3.0f as f32 for cmpf
    // ---------------------------------------------------------------
    $display("Test 5: multi-cmp body");

    // cmpi=eq(0), cmpf=oeq(1): (5==3?)=0, (5.0==3.0?)=0
    @(negedge clk);
    p2_cfg_data = {4'd1, 4'd0};
    @(posedge clk);
    drive_multi_cmp(32'd5, 32'd3, F32_5_0, F32_3_0, 1'b0, 1'b0);

    // cmpi=slt(2), cmpf=ogt(2): (5<3?)=0, (5.0>3.0?)=1
    @(negedge clk);
    p2_cfg_data = {4'd2, 4'd2};
    @(posedge clk);
    drive_multi_cmp(32'd5, 32'd3, F32_5_0, F32_3_0, 1'b0, 1'b1);

    // cmpi=sgt(4), cmpf=olt(4): (5>3?)=1, (5.0<3.0?)=0
    @(negedge clk);
    p2_cfg_data = {4'd4, 4'd4};
    @(posedge clk);
    drive_multi_cmp(32'd5, 32'd3, F32_5_0, F32_3_0, 1'b1, 1'b0);

    // cmpi=ne(1), cmpf=one(6): (5!=3?)=1, (5.0!=3.0?)=1
    @(negedge clk);
    p2_cfg_data = {4'd6, 4'd1};
    @(posedge clk);
    drive_multi_cmp(32'd5, 32'd3, F32_5_0, F32_3_0, 1'b1, 1'b1);
    pass_count = pass_count + 1;

    // ---------------------------------------------------------------
    // Test 6: Runtime predicate switching on p0
    //   Start with slt(2) for (3,5)=1, switch to sgt(4) => 0
    // ---------------------------------------------------------------
    $display("Test 6: runtime predicate switching");
    @(negedge clk);
    p0_cfg_data = 4'd2; // slt
    @(posedge clk);
    drive_cmpi(32'd3, 32'd5, 1'b1); // 3 < 5 = true

    @(negedge clk);
    p0_cfg_data = 4'd4; // sgt
    @(posedge clk);
    drive_cmpi(32'd3, 32'd5, 1'b0); // 3 > 5 = false
    pass_count = pass_count + 1;

    $display("PASS: tb_pe_compare_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : timeout
    #240000;
    $fatal(1, "TIMEOUT");
  end

endmodule
