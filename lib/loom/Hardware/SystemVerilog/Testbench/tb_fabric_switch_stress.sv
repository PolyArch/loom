//===-- tb_fabric_switch_stress.sv - Switch randomized stress test -*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_fabric_switch_stress;

  parameter int NUM_INPUTS  = 4;
  parameter int NUM_OUTPUTS = 4;
  parameter int DATA_WIDTH  = 24;
  parameter int TAG_WIDTH   = 0;
  parameter int NUM_CYCLES  = 320;
  parameter int SEED        = 32'h1357_9BDF;

  localparam int PAYLOAD_WIDTH = DATA_WIDTH + TAG_WIDTH;
  localparam int NUM_CONNECTED = NUM_INPUTS * NUM_OUTPUTS;

  logic clk;
  logic rst_n;

  logic [NUM_INPUTS-1:0]                     in_valid;
  logic [NUM_INPUTS-1:0]                     in_ready;
  logic [NUM_INPUTS-1:0][PAYLOAD_WIDTH-1:0]  in_data;

  logic [NUM_OUTPUTS-1:0]                    out_valid;
  logic [NUM_OUTPUTS-1:0]                    out_ready;
  logic [NUM_OUTPUTS-1:0][PAYLOAD_WIDTH-1:0] out_data;

  logic [NUM_CONNECTED-1:0] cfg_route_table;

  logic        error_valid;
  logic [15:0] error_code;

  fabric_switch #(
    .NUM_INPUTS  (NUM_INPUTS),
    .NUM_OUTPUTS (NUM_OUTPUTS),
    .DATA_WIDTH  (DATA_WIDTH),
    .TAG_WIDTH   (TAG_WIDTH)
  ) dut (
    .clk            (clk),
    .rst_n          (rst_n),
    .in_valid       (in_valid),
    .in_ready       (in_ready),
    .in_data        (in_data),
    .out_valid      (out_valid),
    .out_ready      (out_ready),
    .out_data       (out_data),
    .cfg_route_table(cfg_route_table),
    .error_valid    (error_valid),
    .error_code     (error_code)
  );

  initial begin : clk_gen
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_fabric_switch_stress);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_fabric_switch_stress, "+mda");
  end
`endif

  function automatic int lcg_next(input int state);
    lcg_next = state * 1103515245 + 12345;
  endfunction

  initial begin : main
    int pass_count;
    int handshake_count;
    int rng;
    int iter_var0;
    int iter_var1;
    int iter_var2;
    int route_src [NUM_OUTPUTS];
    int input_routed [NUM_INPUTS];
    int expected_ready;
    logic [NUM_CONNECTED-1:0] bad_route;

    pass_count = 0;
    handshake_count = 0;
    rng = SEED;

    rst_n = 1'b0;
    in_valid = '0;
    in_data = '0;
    out_ready = '0;
    cfg_route_table = '0;

    repeat (3) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    if (error_valid !== 1'b0) begin : check_reset
      $fatal(1, "error_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    for (iter_var0 = 0; iter_var0 < NUM_CYCLES; iter_var0 = iter_var0 + 1) begin : stress_loop
      @(negedge clk);

      // Random legal route: each output picks at most one input.
      cfg_route_table = '0;
      for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1 = iter_var1 + 1) begin : build_route
        rng = lcg_next(rng);
        if (((rng >> 16) & 16'h7) == 0) begin : disable_out
          route_src[iter_var1] = -1;
        end else begin : assign_out
          route_src[iter_var1] = ((rng >> 20) & 16'h7FFF) % NUM_INPUTS;
          cfg_route_table[iter_var1 * NUM_INPUTS + route_src[iter_var1]] = 1'b1;
        end
      end

      // Force occasional broadcast by duplicating one source across two outputs.
      rng = lcg_next(rng);
      if (NUM_OUTPUTS >= 2 && (((rng >> 19) & 16'h3) == 0) && route_src[0] >= 0) begin : make_bcast
        cfg_route_table[1 * NUM_INPUTS +: NUM_INPUTS] = '0;
        route_src[1] = route_src[0];
        cfg_route_table[1 * NUM_INPUTS + route_src[1]] = 1'b1;
      end

      // Build input-routed mask and randomized traffic.
      for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1 = iter_var1 + 1) begin : clear_routed
        input_routed[iter_var1] = 0;
      end
      for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1 = iter_var1 + 1) begin : mark_routed
        if (route_src[iter_var1] >= 0)
          input_routed[route_src[iter_var1]] = 1;
      end

      for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1 = iter_var1 + 1) begin : rand_ready
        rng = lcg_next(rng);
        out_ready[iter_var1] = rng[0];
      end

      for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1 = iter_var1 + 1) begin : rand_input
        rng = lcg_next(rng);
        if (input_routed[iter_var1] != 0)
          in_valid[iter_var1] = rng[0];
        else
          in_valid[iter_var1] = 1'b0;
        rng = lcg_next(rng);
        in_data[iter_var1] = PAYLOAD_WIDTH'(rng);
      end

      #1;

      if (error_valid !== 1'b0) begin : unexpected_err
        $fatal(1, "unexpected error in legal stress loop: code=%0d cycle=%0d",
               error_code, iter_var0);
      end

      // Exact combinational checks for legal route table.
      for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1 = iter_var1 + 1) begin : check_outputs
        if (route_src[iter_var1] >= 0) begin : routed
          if (out_valid[iter_var1] !== in_valid[route_src[iter_var1]]) begin : bad_valid
            $fatal(1, "out_valid mismatch at out%0d cycle=%0d", iter_var1, iter_var0);
          end
          if (out_data[iter_var1] !== in_data[route_src[iter_var1]]) begin : bad_data
            $fatal(1, "out_data mismatch at out%0d cycle=%0d", iter_var1, iter_var0);
          end
        end else begin : unrouted
          if (out_valid[iter_var1] !== 1'b0) begin : bad_unrouted
            $fatal(1, "out_valid should be 0 on unrouted out%0d cycle=%0d", iter_var1, iter_var0);
          end
        end
      end

      for (iter_var1 = 0; iter_var1 < NUM_INPUTS; iter_var1 = iter_var1 + 1) begin : check_ready
        if (input_routed[iter_var1] != 0) begin : routed
          expected_ready = 1;
          for (iter_var2 = 0; iter_var2 < NUM_OUTPUTS; iter_var2 = iter_var2 + 1) begin : and_targets
            if (route_src[iter_var2] == iter_var1)
              expected_ready = expected_ready & int'(out_ready[iter_var2]);
          end
          if (in_ready[iter_var1] !== expected_ready[0]) begin : bad_ready
            $fatal(1, "in_ready mismatch at in%0d cycle=%0d", iter_var1, iter_var0);
          end
        end else begin : unrouted
          if (in_ready[iter_var1] !== 1'b0) begin : bad_unrouted
            $fatal(1, "in_ready should be 0 on unrouted in%0d cycle=%0d", iter_var1, iter_var0);
          end
        end
      end

      for (iter_var1 = 0; iter_var1 < NUM_OUTPUTS; iter_var1 = iter_var1 + 1) begin : count_hs
        if (out_valid[iter_var1] && out_ready[iter_var1])
          handshake_count = handshake_count + 1;
      end

      @(posedge clk);
    end

    if (handshake_count < 20) begin : too_few
      $fatal(1, "stress produced too few handshakes (%0d)", handshake_count);
    end
    pass_count = pass_count + 1;

    // Explicit invalid route check: one output selects two inputs.
    rst_n = 1'b0;
    in_valid = '0;
    out_ready = '0;
    cfg_route_table = '0;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    bad_route = '0;
    bad_route[0 * NUM_INPUTS + 0] = 1'b1;
    bad_route[0 * NUM_INPUTS + 1] = 1'b1;
    cfg_route_table = bad_route;
    @(posedge clk);
    @(posedge clk);

    if (error_valid !== 1'b1 || error_code !== CFG_SWITCH_ROUTE_MIX_INPUTS_TO_SAME_OUTPUT) begin : cfg_error_check
      $fatal(1, "expected CFG_SWITCH_ROUTE_MIX_INPUTS_TO_SAME_OUTPUT, got valid=%0b code=%0d",
             error_valid, error_code);
    end
    pass_count = pass_count + 1;

    // Multi-error precedence: CFG (code 1) + RT (code 262) same cycle.
    // CFG_SWITCH_ROUTE_MIX_INPUTS_TO_SAME_OUTPUT must win (smallest code).
    rst_n = 1'b0;
    in_valid = '0;
    out_ready = '0;
    cfg_route_table = '0;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // Output 0 selects both input 0 and input 1 -> CFG error code 1.
    bad_route = '0;
    bad_route[0 * NUM_INPUTS + 0] = 1'b1;
    bad_route[0 * NUM_INPUTS + 1] = 1'b1;
    cfg_route_table = bad_route;
    // Input 2 is valid but has no route -> RT error code 262.
    in_valid[2] = 1'b1;
    in_data[2] = PAYLOAD_WIDTH'(32'hDEAD);
    @(posedge clk);
    @(posedge clk);

    if (error_valid !== 1'b1) begin : multi_err_valid_check
      $fatal(1, "multi-error: expected error_valid=1");
    end
    if (error_code !== CFG_SWITCH_ROUTE_MIX_INPUTS_TO_SAME_OUTPUT) begin : multi_err_code_check
      $fatal(1, "multi-error: expected code %0d (CFG), got %0d",
             CFG_SWITCH_ROUTE_MIX_INPUTS_TO_SAME_OUTPUT, error_code);
    end
    pass_count = pass_count + 1;

    $display("PASS: tb_fabric_switch_stress (%0d checks, %0d handshakes)",
             pass_count, handshake_count);
    $finish;
  end

  initial begin : timeout
    #300000;
    $fatal(1, "TIMEOUT");
  end

endmodule
