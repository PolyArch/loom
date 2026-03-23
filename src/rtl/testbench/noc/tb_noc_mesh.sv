// tb_noc_mesh.sv -- 2x2 mesh integration testbench.
//
// Instantiates a 2x2 noc_mesh_top and runs an all-to-all traffic
// pattern using tb_noc_traffic_gen instances.  Verifies that all
// injected flits are delivered without deadlock (via a timeout).
//
// Non-synthesizable (testbench only).

`timescale 1ns/1ps

module tb_noc_mesh;
  import noc_pkg::*;

  // ---------------------------------------------------------------
  // Parameters
  // ---------------------------------------------------------------
  localparam int unsigned DATA_WIDTH   = 32;
  localparam int unsigned NUM_VC       = 2;
  localparam int unsigned BUFFER_DEPTH = 4;
  localparam int unsigned MESH_ROWS    = 2;
  localparam int unsigned MESH_COLS    = 2;
  localparam int unsigned NUM_ROUTERS  = MESH_ROWS * MESH_COLS;
  localparam int unsigned FLIT_W       = flit_width(DATA_WIDTH);
  localparam int unsigned FLITS_PER_GEN = 8;
  localparam int unsigned TIMEOUT_CYCLES = 2000;

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
  logic [FLIT_W-1:0] inject_flit  [NUM_ROUTERS];
  logic               inject_valid [NUM_ROUTERS];
  logic               inject_ready [NUM_ROUTERS];

  logic [FLIT_W-1:0] eject_flit   [NUM_ROUTERS];
  logic               eject_valid  [NUM_ROUTERS];
  logic               eject_ready  [NUM_ROUTERS];

  // ---------------------------------------------------------------
  // DUT
  // ---------------------------------------------------------------
  noc_mesh_top #(
    .MESH_ROWS    (MESH_ROWS),
    .MESH_COLS    (MESH_COLS),
    .DATA_WIDTH   (DATA_WIDTH),
    .NUM_VC       (NUM_VC),
    .BUFFER_DEPTH (BUFFER_DEPTH)
  ) u_mesh (
    .clk               (clk),
    .rst_n             (rst_n),
    .core_inject_flit  (inject_flit),
    .core_inject_valid (inject_valid),
    .core_inject_ready (inject_ready),
    .core_eject_flit   (eject_flit),
    .core_eject_valid  (eject_valid),
    .core_eject_ready  (eject_ready)
  );

  // ---------------------------------------------------------------
  // Traffic generators (one per router)
  // ---------------------------------------------------------------
  logic [NUM_ROUTERS-1:0] gen_done;
  logic [31:0]            gen_sent [NUM_ROUTERS];

  generate
    genvar gv_gen;
    for (gv_gen = 0; gv_gen < NUM_ROUTERS; gv_gen = gv_gen + 1) begin : gen_traffic
      tb_noc_traffic_gen #(
        .DATA_WIDTH  (DATA_WIDTH),
        .MESH_ROWS   (MESH_ROWS),
        .MESH_COLS   (MESH_COLS),
        .SRC_ID      (gv_gen),
        .NUM_FLITS   (FLITS_PER_GEN),
        .INJECT_RATE (30),
        .PATTERN     (2)   // all-to-all
      ) u_tgen (
        .clk        (clk),
        .rst_n      (rst_n),
        .flit_out   (inject_flit[gv_gen]),
        .flit_valid (inject_valid[gv_gen]),
        .flit_ready (inject_ready[gv_gen]),
        .done       (gen_done[gv_gen]),
        .sent_count (gen_sent[gv_gen])
      );
    end : gen_traffic
  endgenerate

  // ---------------------------------------------------------------
  // Ejection sink: always accept and count received flits
  // ---------------------------------------------------------------
  logic [31:0] recv_count [NUM_ROUTERS];

  always_comb begin : eject_ready_drive
    integer iter_var0;
    for (iter_var0 = 0; iter_var0 < NUM_ROUTERS; iter_var0 = iter_var0 + 1) begin : eject_rdy_loop
      eject_ready[iter_var0] = 1'b1;
    end : eject_rdy_loop
  end : eject_ready_drive

  always_ff @(posedge clk or negedge rst_n) begin : recv_counter
    integer iter_var0;
    if (!rst_n) begin : recv_reset
      for (iter_var0 = 0; iter_var0 < NUM_ROUTERS; iter_var0 = iter_var0 + 1) begin : recv_reset_loop
        recv_count[iter_var0] <= '0;
      end : recv_reset_loop
    end : recv_reset
    else begin : recv_active
      for (iter_var0 = 0; iter_var0 < NUM_ROUTERS; iter_var0 = iter_var0 + 1) begin : recv_count_loop
        if (eject_valid[iter_var0] && eject_ready[iter_var0]) begin : recv_incr
          recv_count[iter_var0] <= recv_count[iter_var0] + 1;
        end : recv_incr
      end : recv_count_loop
    end : recv_active
  end : recv_counter

  // ---------------------------------------------------------------
  // Test control
  // ---------------------------------------------------------------
  initial begin : main_test
    integer cycle_count;
    integer total_sent;
    integer total_recv;
    integer iter_var0;

    @(posedge rst_n);
    repeat (2) @(posedge clk);

    $display("========================================");
    $display(" tb_noc_mesh: 2x2 all-to-all test");
    $display("========================================");

    // Wait for all generators to finish or timeout.
    cycle_count = 0;
    while (gen_done != {NUM_ROUTERS{1'b1}} && cycle_count < TIMEOUT_CYCLES) begin : wait_loop
      @(posedge clk);
      cycle_count = cycle_count + 1;
    end : wait_loop

    // Allow extra drain time.
    repeat (100) @(posedge clk);

    // Gather totals.
    total_sent = 0;
    total_recv = 0;
    for (iter_var0 = 0; iter_var0 < NUM_ROUTERS; iter_var0 = iter_var0 + 1) begin : sum_loop
      $display("  Router %0d: sent=%0d, recv=%0d",
               iter_var0, gen_sent[iter_var0], recv_count[iter_var0]);
      total_sent = total_sent + int'(gen_sent[iter_var0]);
      total_recv = total_recv + int'(recv_count[iter_var0]);
    end : sum_loop

    $display("\n  Total sent: %0d", total_sent);
    $display("  Total recv: %0d", total_recv);

    if (gen_done != {NUM_ROUTERS{1'b1}}) begin : timeout_fail
      $display("\n  FAIL: Timeout -- not all generators finished (possible deadlock)");
    end : timeout_fail
    else if (total_recv >= total_sent) begin : delivery_pass
      $display("\n  PASS: All flits delivered (%0d cycles)", cycle_count);
    end : delivery_pass
    else begin : delivery_fail
      $display("\n  FAIL: Flit loss detected (sent=%0d, recv=%0d)", total_sent, total_recv);
    end : delivery_fail

    $display("========================================");
    $finish;
  end : main_test

endmodule : tb_noc_mesh
