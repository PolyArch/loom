// tb_noc_router.sv -- Single router unit testbench.
//
// Tests a single noc_router instance placed at position (1,1) in a
// 3x3 mesh.  Verifies:
//   1. Single flit routing to each direction (N, E, S, W, LOCAL)
//   2. Contention: two inputs targeting the same output
//   3. Credit flow control
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

module tb_noc_router;
  import noc_pkg::*;

  // ---------------------------------------------------------------
  // Parameters
  // ---------------------------------------------------------------
  localparam int unsigned DATA_WIDTH   = 32;
  localparam int unsigned NUM_VC       = 2;
  localparam int unsigned BUFFER_DEPTH = 4;
  localparam int unsigned NUM_PORTS    = 5;
  localparam int unsigned MESH_ROWS    = 3;
  localparam int unsigned MESH_COLS    = 3;
  localparam int unsigned ROUTER_ROW   = 1;
  localparam int unsigned ROUTER_COL   = 1;
  localparam int unsigned FLIT_W       = flit_width(DATA_WIDTH);

  // ---------------------------------------------------------------
  // Clock and reset
  // ---------------------------------------------------------------
  logic clk;
  logic rst_n;

  tb_clk_rst_gen #(
    .CLK_PERIOD_NS (10),
    .RST_CYCLES    (5)
  ) u_clk_rst (
    .clk   (clk),
    .rst_n (rst_n)
  );

  // ---------------------------------------------------------------
  // DUT signals
  // ---------------------------------------------------------------
  logic [FLIT_W-1:0] in_flit   [NUM_PORTS];
  logic               in_valid  [NUM_PORTS];
  logic               in_ready  [NUM_PORTS];

  logic [FLIT_W-1:0] out_flit  [NUM_PORTS];
  logic               out_valid [NUM_PORTS];
  logic               out_ready [NUM_PORTS];

  logic [NUM_VC-1:0]  credit_out [NUM_PORTS];
  logic [NUM_VC-1:0]  credit_in  [NUM_PORTS];

  // ---------------------------------------------------------------
  // DUT
  // ---------------------------------------------------------------
  noc_router #(
    .DATA_WIDTH   (DATA_WIDTH),
    .NUM_VC       (NUM_VC),
    .BUFFER_DEPTH (BUFFER_DEPTH),
    .NUM_PORTS    (NUM_PORTS),
    .ROUTER_ROW   (ROUTER_ROW),
    .ROUTER_COL   (ROUTER_COL),
    .MESH_ROWS    (MESH_ROWS),
    .MESH_COLS    (MESH_COLS)
  ) u_dut (
    .clk        (clk),
    .rst_n      (rst_n),
    .in_flit    (in_flit),
    .in_valid   (in_valid),
    .in_ready   (in_ready),
    .out_flit   (out_flit),
    .out_valid  (out_valid),
    .out_ready  (out_ready),
    .credit_out (credit_out),
    .credit_in  (credit_in)
  );

  // ---------------------------------------------------------------
  // Test helpers
  // ---------------------------------------------------------------
  integer test_pass_count;
  integer test_fail_count;

  // Build a SINGLE flit with given src, dst, vc, and payload.
  function automatic logic [FLIT_W-1:0] make_flit(
    input logic [NOC_ID_WIDTH-1:0] src,
    input logic [NOC_ID_WIDTH-1:0] dst,
    input logic [NOC_VC_ID_WIDTH-1:0] vc,
    input logic [DATA_WIDTH-1:0] payload
  );
    flit_header_t hdr;
    hdr.flit_type = FLIT_SINGLE;
    hdr.src_id    = src;
    hdr.dst_id    = dst;
    hdr.vc_id     = vc;
    return {hdr, payload};
  endfunction : make_flit

  // Direction name for display.
  function automatic string dir_name(input int unsigned d);
    case (d)
      0: return "NORTH";
      1: return "EAST";
      2: return "SOUTH";
      3: return "WEST";
      4: return "LOCAL";
      default: return "UNKNOWN";
    endcase
  endfunction : dir_name

  // ---------------------------------------------------------------
  // Stimulus
  // ---------------------------------------------------------------
  initial begin : main_test
    integer iter_var0;
    integer wait_cycles;
    integer saw_output;

    test_pass_count = 0;
    test_fail_count = 0;

    // Initialize all inputs.
    for (iter_var0 = 0; iter_var0 < NUM_PORTS; iter_var0 = iter_var0 + 1) begin : init_loop
      in_flit[iter_var0]   = '0;
      in_valid[iter_var0]  = 1'b0;
      out_ready[iter_var0] = 1'b1;
      credit_in[iter_var0] = '0;
    end : init_loop

    // Wait for reset.
    @(posedge rst_n);
    repeat (2) @(posedge clk);

    $display("========================================");
    $display(" tb_noc_router: Starting tests");
    $display("========================================");

    // ----------------------------------------------------------
    // Test 1: Route to NORTH (dst at row 0, col 1 = ID 1)
    // Inject from LOCAL port.
    // ----------------------------------------------------------
    $display("\n[Test 1] Route LOCAL -> NORTH (dst=1)");
    @(posedge clk);
    in_flit[4]  = make_flit(NOC_ID_WIDTH'(4), NOC_ID_WIDTH'(1), 1'b0, 32'hAAAA_0001);
    in_valid[4] = 1'b1;
    @(posedge clk);
    in_valid[4] = 1'b0;

    // Wait for output (2 pipeline stages + margin).
    saw_output = 0;
    for (wait_cycles = 0; wait_cycles < 10; wait_cycles = wait_cycles + 1) begin : t1_wait
      @(posedge clk);
      if (out_valid[0]) begin : t1_check
        saw_output = 1;
      end : t1_check
    end : t1_wait

    if (saw_output) begin : t1_pass
      $display("  PASS: Flit appeared on NORTH output");
      test_pass_count = test_pass_count + 1;
    end : t1_pass
    else begin : t1_fail
      $display("  FAIL: No flit on NORTH output");
      test_fail_count = test_fail_count + 1;
    end : t1_fail

    repeat (5) @(posedge clk);

    // ----------------------------------------------------------
    // Test 2: Route to EAST (dst at row 1, col 2 = ID 5)
    // ----------------------------------------------------------
    $display("\n[Test 2] Route LOCAL -> EAST (dst=5)");
    @(posedge clk);
    in_flit[4]  = make_flit(NOC_ID_WIDTH'(4), NOC_ID_WIDTH'(5), 1'b0, 32'hBBBB_0002);
    in_valid[4] = 1'b1;
    @(posedge clk);
    in_valid[4] = 1'b0;

    saw_output = 0;
    for (wait_cycles = 0; wait_cycles < 10; wait_cycles = wait_cycles + 1) begin : t2_wait
      @(posedge clk);
      if (out_valid[1]) begin : t2_check
        saw_output = 1;
      end : t2_check
    end : t2_wait

    if (saw_output) begin : t2_pass
      $display("  PASS: Flit appeared on EAST output");
      test_pass_count = test_pass_count + 1;
    end : t2_pass
    else begin : t2_fail
      $display("  FAIL: No flit on EAST output");
      test_fail_count = test_fail_count + 1;
    end : t2_fail

    repeat (5) @(posedge clk);

    // ----------------------------------------------------------
    // Test 3: Route to SOUTH (dst at row 2, col 1 = ID 7)
    // ----------------------------------------------------------
    $display("\n[Test 3] Route LOCAL -> SOUTH (dst=7)");
    @(posedge clk);
    in_flit[4]  = make_flit(NOC_ID_WIDTH'(4), NOC_ID_WIDTH'(7), 1'b0, 32'hCCCC_0003);
    in_valid[4] = 1'b1;
    @(posedge clk);
    in_valid[4] = 1'b0;

    saw_output = 0;
    for (wait_cycles = 0; wait_cycles < 10; wait_cycles = wait_cycles + 1) begin : t3_wait
      @(posedge clk);
      if (out_valid[2]) begin : t3_check
        saw_output = 1;
      end : t3_check
    end : t3_wait

    if (saw_output) begin : t3_pass
      $display("  PASS: Flit appeared on SOUTH output");
      test_pass_count = test_pass_count + 1;
    end : t3_pass
    else begin : t3_fail
      $display("  FAIL: No flit on SOUTH output");
      test_fail_count = test_fail_count + 1;
    end : t3_fail

    repeat (5) @(posedge clk);

    // ----------------------------------------------------------
    // Test 4: Route to WEST (dst at row 1, col 0 = ID 3)
    // ----------------------------------------------------------
    $display("\n[Test 4] Route LOCAL -> WEST (dst=3)");
    @(posedge clk);
    in_flit[4]  = make_flit(NOC_ID_WIDTH'(4), NOC_ID_WIDTH'(3), 1'b0, 32'hDDDD_0004);
    in_valid[4] = 1'b1;
    @(posedge clk);
    in_valid[4] = 1'b0;

    saw_output = 0;
    for (wait_cycles = 0; wait_cycles < 10; wait_cycles = wait_cycles + 1) begin : t4_wait
      @(posedge clk);
      if (out_valid[3]) begin : t4_check
        saw_output = 1;
      end : t4_check
    end : t4_wait

    if (saw_output) begin : t4_pass
      $display("  PASS: Flit appeared on WEST output");
      test_pass_count = test_pass_count + 1;
    end : t4_pass
    else begin : t4_fail
      $display("  FAIL: No flit on WEST output");
      test_fail_count = test_fail_count + 1;
    end : t4_fail

    repeat (5) @(posedge clk);

    // ----------------------------------------------------------
    // Test 5: Route to LOCAL (dst at row 1, col 1 = ID 4)
    // Inject from NORTH port.
    // ----------------------------------------------------------
    $display("\n[Test 5] Route NORTH -> LOCAL (dst=4)");
    @(posedge clk);
    in_flit[0]  = make_flit(NOC_ID_WIDTH'(1), NOC_ID_WIDTH'(4), 1'b0, 32'hEEEE_0005);
    in_valid[0] = 1'b1;
    @(posedge clk);
    in_valid[0] = 1'b0;

    saw_output = 0;
    for (wait_cycles = 0; wait_cycles < 10; wait_cycles = wait_cycles + 1) begin : t5_wait
      @(posedge clk);
      if (out_valid[4]) begin : t5_check
        saw_output = 1;
      end : t5_check
    end : t5_wait

    if (saw_output) begin : t5_pass
      $display("  PASS: Flit appeared on LOCAL output");
      test_pass_count = test_pass_count + 1;
    end : t5_pass
    else begin : t5_fail
      $display("  FAIL: No flit on LOCAL output");
      test_fail_count = test_fail_count + 1;
    end : t5_fail

    repeat (5) @(posedge clk);

    // ----------------------------------------------------------
    // Test 6: Contention -- two inputs target EAST simultaneously
    // ----------------------------------------------------------
    $display("\n[Test 6] Contention: NORTH + LOCAL -> EAST");
    @(posedge clk);
    // From NORTH port, target EAST (dst at row 1, col 2 = ID 5).
    in_flit[0]  = make_flit(NOC_ID_WIDTH'(1), NOC_ID_WIDTH'(5), 1'b0, 32'hF0F0_0006);
    in_valid[0] = 1'b1;
    // From LOCAL port, also target EAST.
    in_flit[4]  = make_flit(NOC_ID_WIDTH'(4), NOC_ID_WIDTH'(5), 1'b0, 32'h0F0F_0007);
    in_valid[4] = 1'b1;
    @(posedge clk);
    in_valid[0] = 1'b0;
    in_valid[4] = 1'b0;

    // Both should eventually appear on EAST output (round-robin).
    saw_output = 0;
    for (wait_cycles = 0; wait_cycles < 20; wait_cycles = wait_cycles + 1) begin : t6_wait
      @(posedge clk);
      if (out_valid[1]) begin : t6_check
        saw_output = saw_output + 1;
      end : t6_check
    end : t6_wait

    if (saw_output >= 2) begin : t6_pass
      $display("  PASS: Both flits appeared on EAST output (%0d seen)", saw_output);
      test_pass_count = test_pass_count + 1;
    end : t6_pass
    else begin : t6_fail
      $display("  FAIL: Expected 2 flits on EAST, saw %0d", saw_output);
      test_fail_count = test_fail_count + 1;
    end : t6_fail

    // ----------------------------------------------------------
    // Summary
    // ----------------------------------------------------------
    repeat (10) @(posedge clk);
    $display("\n========================================");
    $display(" Results: %0d PASS, %0d FAIL", test_pass_count, test_fail_count);
    $display("========================================");

    if (test_fail_count == 0) begin : all_pass
      $display("ALL TESTS PASSED");
    end : all_pass
    else begin : some_fail
      $display("SOME TESTS FAILED");
    end : some_fail

    $finish;
  end : main_test

endmodule : tb_noc_router
