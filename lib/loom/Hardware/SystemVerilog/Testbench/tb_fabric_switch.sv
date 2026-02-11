//===-- tb_fabric_switch.sv - Switch testbench -----------------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Parameterized testbench for fabric_switch. Covers:
//   - Straight routing (diagonal)
//   - Permutation routing (non-diagonal swap)
//   - Valid/ready handshake with backpressure
//   - Randomized route-table with data verification
//   - CFG_ error injection (ROUTE_MIX_INPUTS_TO_SAME_OUTPUT)
//   - RT_ error injection (UNROUTED_INPUT)
//   - Broadcast: one input to two outputs
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

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_fabric_switch);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_fabric_switch, "+mda");
  end
`endif

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------
  int rng;

  // Build compressed route table from full [NUM_OUTPUTS][NUM_INPUTS] matrix
  function automatic logic [NUM_CONNECTED-1:0] compress_route(
      input logic [NUM_OUTPUTS*NUM_INPUTS-1:0] flat_matrix
  );
    int iter_var0, iter_var1;
    automatic int bit_idx = 0;
    automatic logic [NUM_CONNECTED-1:0] result = '0;
    for (iter_var0 = 0; iter_var0 < NUM_OUTPUTS; iter_var0++) begin : per_out
      for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1++) begin : per_in
        if (CONNECTIVITY[iter_var0*NUM_INPUTS + iter_var1]) begin : connected
          result[bit_idx] = flat_matrix[iter_var0*NUM_INPUTS + iter_var1];
          bit_idx++;
        end
      end
    end
    return result;
  endfunction

  // LCG step: returns next rng state
  function automatic int lcg_next(input int state);
    return state * 1103515245 + 12345;
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
  initial begin : main
    int iter_var0, iter_var1, iter_var2;
    rng = SEED + 1;

    do_reset();

    // ---- Test 1: Post-reset state ----
    if (error_valid !== 0)
      $fatal(1, "FAIL: error_valid should be 0 after reset");

    // ---- Test 2: Straight routing (input k -> output k for k < DIM) ----
    begin : straight_route
      logic [NUM_OUTPUTS*NUM_INPUTS-1:0] flat_route;
      flat_route = '0;
      for (iter_var0 = 0; iter_var0 < DIM; iter_var0++) begin : set_diag
        flat_route[iter_var0*NUM_INPUTS + iter_var0] = 1'b1;
      end

      cfg_route_table = compress_route(flat_route);
      out_ready = '1;
      #1;

      for (iter_var0 = 0; iter_var0 < DIM; iter_var0++) begin : check_out
        in_valid = '0;
        in_valid[iter_var0] = 1'b1;
        in_data[iter_var0] = PAYLOAD_WIDTH'(iter_var0 + 1);
        #1;
        if (out_valid[iter_var0] !== 1'b1)
          $fatal(1, "FAIL: straight routing: out_valid[%0d] should be 1", iter_var0);
        if (out_data[iter_var0] !== PAYLOAD_WIDTH'(iter_var0 + 1))
          $fatal(1, "FAIL: straight routing: data mismatch at output %0d", iter_var0);
      end
      in_valid = '0;
    end

    do_reset();

    // ---- Test 3: Valid/ready handshake with backpressure ----
    if (NUM_INPUTS >= 2 && NUM_OUTPUTS >= 2) begin : backpressure
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

    // ---- Test 4: Permutation routing (non-diagonal swap) ----
    // Route input 0 -> output (DIM-1), input 1 -> output (DIM-2), ...
    // This is a reversed mapping that verifies non-diagonal route correctness.
    if (DIM >= 2) begin : permutation
      logic [NUM_OUTPUTS*NUM_INPUTS-1:0] flat_route;
      flat_route = '0;
      for (iter_var0 = 0; iter_var0 < DIM; iter_var0++) begin : set_perm
        flat_route[(DIM-1-iter_var0)*NUM_INPUTS + iter_var0] = 1'b1;
      end

      cfg_route_table = compress_route(flat_route);
      out_ready = '1;

      // Drive all DIM inputs simultaneously with unique data
      in_valid = '0;
      for (iter_var0 = 0; iter_var0 < DIM; iter_var0++) begin : drive_perm
        in_valid[iter_var0] = 1'b1;
        in_data[iter_var0] = PAYLOAD_WIDTH'(16'hC000 + iter_var0);
      end
      #1;

      // Verify: output[DIM-1-k] should have data from input[k]
      for (iter_var0 = 0; iter_var0 < DIM; iter_var0++) begin : check_perm
        if (out_valid[DIM-1-iter_var0] !== 1'b1)
          $fatal(1, "FAIL: permutation: out_valid[%0d] should be 1", DIM-1-iter_var0);
        if (out_data[DIM-1-iter_var0] !== PAYLOAD_WIDTH'(16'hC000 + iter_var0))
          $fatal(1, "FAIL: permutation: out_data[%0d] expected 0x%04x, got 0x%04x",
                 DIM-1-iter_var0, 16'hC000 + iter_var0, out_data[DIM-1-iter_var0]);
      end
      in_valid = '0;
    end

    do_reset();

    // ---- Test 5: Randomized route-table with data verification ----
    // Each transaction: generate a random valid one-to-one mapping under
    // connectivity constraints, drive random data, verify output correctness.
    begin : rand_route
      logic [NUM_OUTPUTS*NUM_INPUTS-1:0] flat_route;
      logic [NUM_INPUTS-1:0] input_used;
      int mapping [NUM_OUTPUTS];
      int candidates [32];
      int n_cand, pick, chosen;

      for (iter_var0 = 0; iter_var0 < NUM_TRANSACTIONS; iter_var0++) begin : txn
        // Build a random valid one-to-one route mapping
        flat_route = '0;
        input_used = '0;

        for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1++) begin : build_map
          mapping[iter_var1] = -1;
          n_cand = 0;
          for (iter_var2 = 0; iter_var2 < NUM_INPUTS; iter_var2++) begin : find_cand
            if (CONNECTIVITY[iter_var1*NUM_INPUTS + iter_var2] && !input_used[iter_var2]) begin : add_cand
              candidates[n_cand] = iter_var2;
              n_cand++;
            end
          end
          if (n_cand > 0) begin : pick_route
            rng = lcg_next(rng);
            pick = ((rng >> 16) & 32'h7FFF) % n_cand;
            chosen = candidates[pick];
            flat_route[iter_var1*NUM_INPUTS + chosen] = 1'b1;
            input_used[chosen] = 1'b1;
            mapping[iter_var1] = chosen;
          end
        end

        cfg_route_table = compress_route(flat_route);
        out_ready = '1;

        // Drive random data on routed inputs
        in_valid = '0;
        for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1++) begin : drive_in
          in_valid[iter_var1] = input_used[iter_var1];
          rng = lcg_next(rng);
          in_data[iter_var1] = PAYLOAD_WIDTH'(rng);
        end
        #1;

        // Verify each routed output
        for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1++) begin : check_out
          if (mapping[iter_var1] >= 0) begin : routed
            if (out_valid[iter_var1] !== 1'b1)
              $fatal(1, "FAIL: random route t=%0d: out_valid[%0d] not 1", iter_var0, iter_var1);
            if (out_data[iter_var1] !== in_data[mapping[iter_var1]])
              $fatal(1, "FAIL: random route t=%0d: data mismatch out[%0d], expected from in[%0d]",
                     iter_var0, iter_var1, mapping[iter_var1]);
          end else begin : unrouted
            if (out_valid[iter_var1] !== 1'b0)
              $fatal(1, "FAIL: random route t=%0d: out_valid[%0d] should be 0 (unrouted)", iter_var0, iter_var1);
          end
        end

        @(posedge clk);
        #1;
      end
      in_valid = '0;
    end

    do_reset();

    // ---- Test 6: CFG_SWITCH_ROUTE_MIX_INPUTS_TO_SAME_OUTPUT (code 1) ----
    if (NUM_INPUTS >= 2) begin : mix_inputs
      logic [NUM_OUTPUTS*NUM_INPUTS-1:0] flat_route;
      flat_route = '0;
      flat_route[0*NUM_INPUTS + 0] = 1'b1;
      flat_route[0*NUM_INPUTS + 1] = 1'b1;

      cfg_route_table = compress_route(flat_route);
      @(posedge clk);
      @(posedge clk);
      #1;

      if (error_valid !== 1'b1)
        $fatal(1, "FAIL: CFG_SWITCH_ROUTE_MIX_INPUTS_TO_SAME_OUTPUT: error_valid should be 1");
      if (error_code !== 16'd1)
        $fatal(1, "FAIL: CFG_SWITCH_ROUTE_MIX_INPUTS_TO_SAME_OUTPUT: error_code should be 1, got %0d",
               error_code);
    end

    do_reset();

    // ---- Test 7: Broadcast (one input to two outputs) ----
    if (NUM_OUTPUTS >= 2) begin : broadcast
      logic [NUM_OUTPUTS*NUM_INPUTS-1:0] flat_route;
      flat_route = '0;
      flat_route[0*NUM_INPUTS + 0] = 1'b1;  // out0 <- in0
      flat_route[1*NUM_INPUTS + 0] = 1'b1;  // out1 <- in0 (broadcast)

      cfg_route_table = compress_route(flat_route);
      out_ready = '1;
      in_valid = '0;
      in_valid[0] = 1'b1;
      in_data[0] = PAYLOAD_WIDTH'(16'hBEEF);
      #1;

      // No error expected (broadcast is valid)
      if (error_valid !== 1'b0)
        $fatal(1, "FAIL: broadcast: unexpected error code %0d", error_code);
      // Both outputs should receive the data
      if (out_valid[0] !== 1'b1)
        $fatal(1, "FAIL: broadcast: out_valid[0] should be 1");
      if (out_valid[1] !== 1'b1)
        $fatal(1, "FAIL: broadcast: out_valid[1] should be 1");
      if (out_data[0] !== PAYLOAD_WIDTH'(16'hBEEF))
        $fatal(1, "FAIL: broadcast: out_data[0] mismatch");
      if (out_data[1] !== PAYLOAD_WIDTH'(16'hBEEF))
        $fatal(1, "FAIL: broadcast: out_data[1] mismatch");
      // in_ready should be 1 (both out_ready are 1)
      if (in_ready[0] !== 1'b1)
        $fatal(1, "FAIL: broadcast: in_ready[0] should be 1 when all targets ready");

      // Now test backpressure: deassert out_ready[1]
      out_ready[1] = 1'b0;
      #1;
      // in_ready should be 0 (not all broadcast targets ready)
      if (in_ready[0] !== 1'b0)
        $fatal(1, "FAIL: broadcast backpressure: in_ready[0] should be 0 when out_ready[1]=0");
      // out_valid[0] should still be 1 (valid does not depend on ready)
      if (out_valid[0] !== 1'b1)
        $fatal(1, "FAIL: broadcast backpressure: out_valid[0] should be 1 (valid independent of ready)");

      in_valid = '0;
      out_ready = '0;
    end

    do_reset();

    // ---- Test 8: RT_SWITCH_UNROUTED_INPUT (code 262) ----
    begin : unrouted_input
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
  initial begin : watchdog
    #(NUM_TRANSACTIONS * 100 * 10 + 100000);
    $fatal(1, "FAIL: testbench watchdog timeout");
  end

endmodule
