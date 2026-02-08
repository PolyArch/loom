//===-- tb_fabric_switch.sv - Switch testbench -----------------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Parameterized testbench for fabric_switch. Covers:
//   - Straight routing (diagonal)
//   - Valid/ready handshake with backpressure
//   - Randomized valid routing
//   - CFG_ error injection (ROUTE_MULTI_OUT, ROUTE_MULTI_IN)
//   - RT_ error injection (UNROUTED_INPUT)
//
//===----------------------------------------------------------------------===//

module tb_fabric_switch #(
    parameter int NUM_INPUTS       = 2,
    parameter int NUM_OUTPUTS      = 2,
    parameter int DATA_WIDTH       = 32,
    parameter int TAG_WIDTH        = 0,
    parameter bit [NUM_OUTPUTS*NUM_INPUTS-1:0] CONNECTIVITY = '1,
    parameter int NUM_TRANSACTIONS = 100,
    parameter int SEED             = 0
);

  localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH;
  localparam int NUM_CONNECTED = $countones(CONNECTIVITY);
  localparam int DIM = (NUM_INPUTS < NUM_OUTPUTS) ? NUM_INPUTS : NUM_OUTPUTS;

  logic clk;
  logic rst_n;

  logic [NUM_INPUTS-1:0]                     in_valid;
  logic [NUM_INPUTS-1:0]                     in_ready;
  logic [NUM_INPUTS-1:0][PAYLOAD_WIDTH-1:0]  in_data;

  logic [NUM_OUTPUTS-1:0]                    out_valid;
  logic [NUM_OUTPUTS-1:0]                    out_ready;
  logic [NUM_OUTPUTS-1:0][PAYLOAD_WIDTH-1:0] out_data;

  logic [NUM_CONNECTED-1:0]                  cfg_route_table;

  logic        error_valid;
  logic [15:0] error_code;

  fabric_switch #(
    .NUM_INPUTS   (NUM_INPUTS),
    .NUM_OUTPUTS  (NUM_OUTPUTS),
    .DATA_WIDTH   (DATA_WIDTH),
    .TAG_WIDTH    (TAG_WIDTH),
    .CONNECTIVITY (CONNECTIVITY)
  ) dut (
    .clk             (clk),
    .rst_n           (rst_n),
    .in_valid        (in_valid),
    .in_ready        (in_ready),
    .in_data         (in_data),
    .out_valid       (out_valid),
    .out_ready       (out_ready),
    .out_data        (out_data),
    .cfg_route_table (cfg_route_table),
    .error_valid     (error_valid),
    .error_code      (error_code)
  );

  // Clock generation
  initial clk = 0;
  always #5 clk = ~clk;

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------
  int rng;

  // Build compressed route table from full [NUM_OUTPUTS][NUM_INPUTS] matrix
  function automatic logic [NUM_CONNECTED-1:0] compress_route(
      input logic [NUM_OUTPUTS*NUM_INPUTS-1:0] flat_matrix
  );
    automatic int bit_idx = 0;
    automatic logic [NUM_CONNECTED-1:0] result = '0;
    for (int o = 0; o < NUM_OUTPUTS; o++)
      for (int i = 0; i < NUM_INPUTS; i++)
        if (CONNECTIVITY[o*NUM_INPUTS + i]) begin
          result[bit_idx] = flat_matrix[o*NUM_INPUTS + i];
          bit_idx++;
        end
    return result;
  endfunction

  task automatic do_reset();
    rst_n = 0;
    in_valid = '0;
    out_ready = '0;
    in_data = '0;
    cfg_route_table = '0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    #1;
  endtask

  // -----------------------------------------------------------------------
  // Main test sequence
  // -----------------------------------------------------------------------
  initial begin
    rng = SEED + 1;

    do_reset();

    // ---- Test 1: Post-reset state ----
    if (error_valid !== 0)
      $fatal(1, "FAIL: error_valid should be 0 after reset");

    // ---- Test 2: Straight routing (input k -> output k for k < DIM) ----
    begin
      logic [NUM_OUTPUTS*NUM_INPUTS-1:0] flat_route;
      flat_route = '0;
      for (int k = 0; k < DIM; k++)
        flat_route[k*NUM_INPUTS + k] = 1'b1;

      cfg_route_table = compress_route(flat_route);
      out_ready = '1;
      #1;

      for (int k = 0; k < DIM; k++) begin
        in_valid = '0;
        in_valid[k] = 1'b1;
        in_data[k] = PAYLOAD_WIDTH'(k + 1);
        #1;
        if (out_valid[k] !== 1'b1)
          $fatal(1, "FAIL: straight routing: out_valid[%0d] should be 1", k);
        if (out_data[k] !== PAYLOAD_WIDTH'(k + 1))
          $fatal(1, "FAIL: straight routing: data mismatch at output %0d", k);
      end
      in_valid = '0;
    end

    do_reset();

    // ---- Test 3: Valid/ready handshake with backpressure ----
    if (NUM_INPUTS >= 2 && NUM_OUTPUTS >= 2) begin
      logic [NUM_OUTPUTS*NUM_INPUTS-1:0] flat_route;
      flat_route = '0;
      flat_route[0*NUM_INPUTS + 0] = 1'b1;  // out0 <- in0
      flat_route[1*NUM_INPUTS + 1] = 1'b1;  // out1 <- in1

      cfg_route_table = compress_route(flat_route);
      in_valid = '0;
      in_valid[0] = 1'b1;
      in_valid[1] = 1'b1;
      in_data[0] = PAYLOAD_WIDTH'(16'hAA);
      in_data[1] = PAYLOAD_WIDTH'(16'hBB);

      out_ready = '0;
      out_ready[1] = 1'b1;
      #1;

      if (in_ready[0] !== 0)
        $fatal(1, "FAIL: backpressure: in_ready[0] should be 0 when out_ready[0]=0");
      if (in_ready[1] !== 1)
        $fatal(1, "FAIL: backpressure: in_ready[1] should be 1 when out_ready[1]=1");

      in_valid = '0;
      out_ready = '0;
    end

    do_reset();

    // ---- Test 4: Randomized valid routing ----
    begin
      logic [NUM_OUTPUTS*NUM_INPUTS-1:0] flat_route;
      flat_route = '0;
      for (int k = 0; k < DIM; k++)
        flat_route[k*NUM_INPUTS + k] = 1'b1;

      cfg_route_table = compress_route(flat_route);
      out_ready = '1;

      for (int t = 0; t < NUM_TRANSACTIONS; t++) begin
        for (int i = 0; i < NUM_INPUTS; i++) begin
          in_valid[i] = (i < DIM) ? 1'b1 : 1'b0;
          rng = rng * 1103515245 + 12345;
          in_data[i] = PAYLOAD_WIDTH'(rng);
        end
        #1;

        for (int k = 0; k < DIM; k++) begin
          if (out_valid[k] !== 1'b1)
            $fatal(1, "FAIL: random route t=%0d: out_valid[%0d] not 1", t, k);
          if (out_data[k] !== in_data[k])
            $fatal(1, "FAIL: random route t=%0d: data mismatch at output %0d", t, k);
        end

        @(posedge clk);
        #1;
      end
      in_valid = '0;
    end

    do_reset();

    // ---- Test 5: CFG_SWITCH_ROUTE_MULTI_OUT (code 1) ----
    if (NUM_INPUTS >= 2) begin
      logic [NUM_OUTPUTS*NUM_INPUTS-1:0] flat_route;
      flat_route = '0;
      flat_route[0*NUM_INPUTS + 0] = 1'b1;
      flat_route[0*NUM_INPUTS + 1] = 1'b1;

      cfg_route_table = compress_route(flat_route);
      @(posedge clk);
      @(posedge clk);
      #1;

      if (error_valid !== 1'b1)
        $fatal(1, "FAIL: CFG_SWITCH_ROUTE_MULTI_OUT: error_valid should be 1");
      if (error_code !== 16'd1)
        $fatal(1, "FAIL: CFG_SWITCH_ROUTE_MULTI_OUT: error_code should be 1, got %0d",
               error_code);
    end

    do_reset();

    // ---- Test 6: CFG_SWITCH_ROUTE_MULTI_IN (code 2) ----
    if (NUM_OUTPUTS >= 2) begin
      logic [NUM_OUTPUTS*NUM_INPUTS-1:0] flat_route;
      flat_route = '0;
      flat_route[0*NUM_INPUTS + 0] = 1'b1;
      flat_route[1*NUM_INPUTS + 0] = 1'b1;

      cfg_route_table = compress_route(flat_route);
      @(posedge clk);
      @(posedge clk);
      #1;

      if (error_valid !== 1'b1)
        $fatal(1, "FAIL: CFG_SWITCH_ROUTE_MULTI_IN: error_valid should be 1");
      if (error_code !== 16'd2)
        $fatal(1, "FAIL: CFG_SWITCH_ROUTE_MULTI_IN: error_code should be 2, got %0d",
               error_code);
    end

    do_reset();

    // ---- Test 7: RT_SWITCH_UNROUTED_INPUT (code 262) ----
    begin
      cfg_route_table = '0;
      in_valid = '0;
      in_valid[0] = 1'b1;
      in_data[0] = PAYLOAD_WIDTH'(16'hDEAD);
      @(posedge clk);
      @(posedge clk);
      #1;

      if (error_valid !== 1'b1)
        $fatal(1, "FAIL: RT_SWITCH_UNROUTED_INPUT: error_valid should be 1");
      if (error_code !== 16'd262)
        $fatal(1, "FAIL: RT_SWITCH_UNROUTED_INPUT: error_code should be 262, got %0d",
               error_code);
    end

    $display("PASS: tb_fabric_switch NUM_INPUTS=%0d NUM_OUTPUTS=%0d DATA_WIDTH=%0d",
             NUM_INPUTS, NUM_OUTPUTS, DATA_WIDTH);
    $finish;
  end

  // Watchdog timer
  initial begin
    #(NUM_TRANSACTIONS * 100 * 10 + 100000);
    $fatal(1, "FAIL: testbench watchdog timeout");
  end

endmodule
