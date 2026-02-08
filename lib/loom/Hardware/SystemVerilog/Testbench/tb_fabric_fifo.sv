//===-- tb_fabric_fifo.sv - FIFO testbench --------------------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Parameterized testbench for fabric_fifo. Covers:
//   - Reset behavior
//   - Single push/pop
//   - Fill to capacity / drain
//   - Randomized traffic (valid/ready)
//   - Bypass mode (when BYPASSABLE=1)
//
//===----------------------------------------------------------------------===//

module tb_fabric_fifo #(
    parameter int DEPTH            = 1,
    parameter int DATA_WIDTH       = 32,
    parameter int TAG_WIDTH        = 0,
    parameter bit BYPASSABLE       = 0,
    parameter int NUM_TRANSACTIONS = 100,
    parameter int SEED             = 0
);

  localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH;
  localparam int CONFIG_WIDTH  = BYPASSABLE ? 1 : 0;

  logic                      clk;
  logic                      rst_n;
  logic                      in_valid;
  logic                      in_ready;
  logic [PAYLOAD_WIDTH-1:0]  in_data;
  logic                      out_valid;
  logic                      out_ready;
  logic [PAYLOAD_WIDTH-1:0]  out_data;
  logic [CONFIG_WIDTH > 0 ? CONFIG_WIDTH-1 : 0 : 0] cfg_data;

  fabric_fifo #(
    .DEPTH      (DEPTH),
    .DATA_WIDTH (DATA_WIDTH),
    .TAG_WIDTH  (TAG_WIDTH),
    .BYPASSABLE (BYPASSABLE)
  ) dut (
    .clk       (clk),
    .rst_n     (rst_n),
    .in_valid  (in_valid),
    .in_ready  (in_ready),
    .in_data   (in_data),
    .out_valid (out_valid),
    .out_ready (out_ready),
    .out_data  (out_data),
    .cfg_data  (cfg_data)
  );

  // Clock generation: period 10ns
  initial clk = 0;
  always #5 clk = ~clk;

  // Storage for randomized test
  int rand_val;
  logic [PAYLOAD_WIDTH-1:0] sent_data [NUM_TRANSACTIONS];
  logic [PAYLOAD_WIDTH-1:0] recv_data [NUM_TRANSACTIONS];
  int sent_count;
  int recv_count;

  initial begin
    rand_val  = SEED + 1;
    in_valid  = 0;
    out_ready = 0;
    in_data   = '0;
    cfg_data  = '0;

    // ---- Reset ----
    rst_n = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    #1;

    // ---- Test 1: Post-reset state ----
    if (out_valid !== 0)
      $fatal(1, "FAIL: out_valid should be 0 after reset");
    if (in_ready !== 1)
      $fatal(1, "FAIL: in_ready should be 1 after reset (empty FIFO)");

    // ---- Test 2: Single push then pop ----
    in_valid = 1;
    in_data  = PAYLOAD_WIDTH'(42);
    @(posedge clk);
    #1;
    in_valid = 0;
    if (out_valid !== 1)
      $fatal(1, "FAIL: after push: out_valid should be 1");
    if (out_data !== PAYLOAD_WIDTH'(42))
      $fatal(1, "FAIL: after push: data mismatch: expected 42 got %0h", out_data);

    out_ready = 1;
    @(posedge clk);
    #1;
    out_ready = 0;
    if (out_valid !== 0)
      $fatal(1, "FAIL: after pop: out_valid should be 0");

    // ---- Test 3: Fill to capacity ----
    for (int i = 0; i < DEPTH; i++) begin
      in_valid = 1;
      in_data  = PAYLOAD_WIDTH'(i + 100);
      @(posedge clk);
      #1;
    end
    in_valid = 0;
    if (in_ready !== 0)
      $fatal(1, "FAIL: in_ready should be 0 when FIFO is full");

    // ---- Test 4: Drain and verify order ----
    for (int i = 0; i < DEPTH; i++) begin
      if (out_valid !== 1)
        $fatal(1, "FAIL: drain: out_valid should be 1 at %0d", i);
      if (out_data !== PAYLOAD_WIDTH'(i + 100))
        $fatal(1, "FAIL: drain mismatch at %0d: expected=%0h got=%0h",
               i, PAYLOAD_WIDTH'(i + 100), out_data);
      out_ready = 1;
      @(posedge clk);
      #1;
      out_ready = 0;
    end
    if (out_valid !== 0)
      $fatal(1, "FAIL: out_valid should be 0 after drain");

    // ---- Test 5: Randomized push/pop traffic ----
    // Reset to clean state
    rst_n = 0;
    in_valid = 0;
    out_ready = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    #1;

    sent_count = 0;
    recv_count = 0;

    for (int i = 0; i < NUM_TRANSACTIONS; i++) begin
      rand_val = rand_val * 1103515245 + 12345;
      sent_data[i] = PAYLOAD_WIDTH'(rand_val);
    end

    // Protocol:
    //   1. @(posedge clk) -> DUT FFs update
    //   2. #1 -> combinational outputs settle with new FF state
    //   3. Drive new stimulus (in_valid, in_data, out_ready)
    //   4. #1 -> combinational outputs settle with new FF state + new inputs
    //   5. Sample handshake (in_valid && in_ready, out_valid && out_ready)
    //   6. Go to step 1: @(posedge clk) latches the handshake
    for (int cycle = 0; cycle < NUM_TRANSACTIONS * 20 + 200; cycle++) begin
      if (recv_count >= NUM_TRANSACTIONS)
        break;

      // Advance to clock edge (latches previous handshake into FFs)
      @(posedge clk);
      // Let FFs settle (DUT updates count, head, tail, buffer)
      #1;

      // Drive new stimulus
      if (sent_count < NUM_TRANSACTIONS) begin
        rand_val = rand_val * 1103515245 + 12345;
        in_valid = ((rand_val >> 16) & 3) != 0;
        in_data  = sent_data[sent_count];
      end else begin
        in_valid = 0;
      end
      rand_val = rand_val * 1103515245 + 12345;
      out_ready = ((rand_val >> 16) & 3) != 0;

      // Let combinational logic settle with new inputs
      #1;

      // Sample handshake: these signal levels will be latched at the
      // next posedge, so this is the correct time to observe them.
      if (in_valid && in_ready && sent_count < NUM_TRANSACTIONS)
        sent_count = sent_count + 1;
      if (out_valid && out_ready && recv_count < NUM_TRANSACTIONS) begin
        recv_data[recv_count] = out_data;
        recv_count = recv_count + 1;
      end
    end

    in_valid  = 0;
    out_ready = 0;

    if (recv_count < NUM_TRANSACTIONS)
      $fatal(1, "FAIL: random traffic: only received %0d/%0d", recv_count, NUM_TRANSACTIONS);

    for (int i = 0; i < NUM_TRANSACTIONS; i++) begin
      if (recv_data[i] !== sent_data[i])
        $fatal(1, "FAIL: random traffic mismatch at %0d: expected=%0h got=%0h",
               i, sent_data[i], recv_data[i]);
    end

    // ---- Test 6: Bypass mode (if BYPASSABLE) ----
    if (BYPASSABLE) begin
      rst_n = 0;
      in_valid = 0;
      out_ready = 0;
      repeat (3) @(posedge clk);
      rst_n = 1;
      @(posedge clk);

      cfg_data = 1'b1;
      @(posedge clk);
      #1;

      in_valid  = 1;
      out_ready = 1;
      in_data   = PAYLOAD_WIDTH'(16'hBEEF);
      #1;

      if (out_valid !== 1)
        $fatal(1, "FAIL: bypass mode: out_valid should be 1");
      if (out_data !== PAYLOAD_WIDTH'(16'hBEEF))
        $fatal(1, "FAIL: bypass mode: data mismatch");
      if (in_ready !== 1)
        $fatal(1, "FAIL: bypass mode: in_ready should follow out_ready");

      out_ready = 0;
      #1;
      if (in_ready !== 0)
        $fatal(1, "FAIL: bypass backpressure: in_ready should be 0");

      in_valid  = 0;
      cfg_data  = '0;
      @(posedge clk);
    end

    $display("PASS: tb_fabric_fifo DEPTH=%0d DATA_WIDTH=%0d TAG_WIDTH=%0d BYPASSABLE=%0d",
             DEPTH, DATA_WIDTH, TAG_WIDTH, BYPASSABLE);
    $finish;
  end

  // Watchdog timer
  initial begin
    #(NUM_TRANSACTIONS * 500 * 10 + 200000);
    $fatal(1, "FAIL: testbench watchdog timeout");
  end

endmodule
