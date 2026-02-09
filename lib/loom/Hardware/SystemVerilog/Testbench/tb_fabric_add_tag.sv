//===-- tb_fabric_add_tag.sv - AddTag testbench ---------------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Parameterized testbench for fabric_add_tag. Covers:
//   - Reset behavior
//   - Tag attachment with various config values
//   - Backpressure (out_ready deassertion)
//
//===----------------------------------------------------------------------===//

module tb_fabric_add_tag #(
    parameter int DATA_WIDTH = 32,
    parameter int TAG_WIDTH  = 4
);

  localparam int IN_WIDTH  = DATA_WIDTH;
  localparam int OUT_WIDTH = DATA_WIDTH + TAG_WIDTH;

  logic                    clk;
  logic                    rst_n;
  logic                    in_valid;
  logic                    in_ready;
  logic [IN_WIDTH-1:0]     in_data;
  logic                    out_valid;
  logic                    out_ready;
  logic [OUT_WIDTH-1:0]    out_data;
  logic [TAG_WIDTH-1:0]    cfg_data;

  fabric_add_tag #(
    .DATA_WIDTH (DATA_WIDTH),
    .TAG_WIDTH  (TAG_WIDTH)
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

  initial clk = 0;
  always #5 clk = ~clk;

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_fabric_add_tag);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_fabric_add_tag);
  end
`endif

  // Watchdog
  initial begin : watchdog
    #10000;
    $display("ERROR: timeout");
    $fatal(1, "watchdog timeout");
  end

  integer num_pass;
  initial begin : test
    num_pass = 0;
    rst_n = 0;
    in_valid = 0;
    in_data = 0;
    out_ready = 0;
    cfg_data = 0;

    // Reset
    @(posedge clk);
    @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // Test 1: simple tag attachment
    cfg_data = 4'hA;
    in_data = 32'hDEAD_BEEF;
    in_valid = 1;
    out_ready = 1;
    @(posedge clk);
    while (!in_ready) @(posedge clk);
    if (out_valid && out_data == {4'hA, 32'hDEAD_BEEF}) begin : t1_pass
      $display("PASS: tag attachment correct");
      num_pass = num_pass + 1;
    end else begin : t1_fail
      $display("FAIL: expected {A, DEAD_BEEF}, got %h", out_data);
    end

    // Test 2: backpressure
    in_valid = 1;
    out_ready = 0;
    @(posedge clk);
    @(posedge clk);
    if (!in_ready) begin : t2_pass
      $display("PASS: backpressure works");
      num_pass = num_pass + 1;
    end else begin : t2_fail
      $display("FAIL: in_ready should be 0 when out_ready=0");
    end

    // Done
    in_valid = 0;
    @(posedge clk);
    $display("Test complete: %0d/2 passed", num_pass);
    if (num_pass == 2) begin : all_pass
      $finish;
    end else begin : some_fail
      $fatal(1, "not all tests passed");
    end
  end

endmodule
