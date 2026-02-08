//===-- tb_switch_top.sv - Smoke test for exported switch_top --*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Minimal smoke testbench for the generated switch_top module.
// Validates: reset state, error port, handshake presence.
//
//===----------------------------------------------------------------------===//

module tb_switch_top;

  logic        clk;
  logic        rst_n;
  logic        in0_valid, in0_ready;
  logic [31:0] in0_data;
  logic        in1_valid, in1_ready;
  logic [31:0] in1_data;
  logic        out0_valid, out0_ready;
  logic [31:0] out0_data;
  logic        out1_valid, out1_ready;
  logic [31:0] out1_data;
  logic        error_valid;
  logic [15:0] error_code;
  logic [3:0]  sw0_cfg_route_table;

  switch_top dut (
    .clk                    (clk),
    .rst_n                  (rst_n),
    .in0_valid              (in0_valid),
    .in0_ready              (in0_ready),
    .in0_data               (in0_data),
    .in1_valid              (in1_valid),
    .in1_ready              (in1_ready),
    .in1_data               (in1_data),
    .out0_valid             (out0_valid),
    .out0_ready             (out0_ready),
    .out0_data              (out0_data),
    .out1_valid             (out1_valid),
    .out1_ready             (out1_ready),
    .out1_data              (out1_data),
    .sw0_cfg_route_table    (sw0_cfg_route_table),
    .error_valid            (error_valid),
    .error_code             (error_code)
  );

  initial clk = 0;
  always #5 clk = ~clk;

  initial begin
    in0_valid  = 0;
    in1_valid  = 0;
    out0_ready = 0;
    out1_ready = 0;
    in0_data   = '0;
    in1_data   = '0;
    sw0_cfg_route_table = '0;

    // Reset
    rst_n = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    #1;

    // Post-reset: error_valid=0
    if (error_valid !== 0)
      $fatal(1, "FAIL: error_valid should be 0 after reset");

    // With route_table='0, no inputs are routed
    // so out_valid should all be 0
    if (out0_valid !== 0)
      $fatal(1, "FAIL: out0_valid should be 0 with no route");
    if (out1_valid !== 0)
      $fatal(1, "FAIL: out1_valid should be 0 with no route");

    // Configure diagonal route: in0->out0, in1->out1
    // For 2x2 full connectivity, route_table bits are:
    //   bit 0: out0<-in0, bit 1: out0<-in1, bit 2: out1<-in0, bit 3: out1<-in1
    sw0_cfg_route_table = 4'b1001;
    out0_ready = 1;
    out1_ready = 1;
    in0_valid  = 1;
    in0_data   = 32'hA0A0;
    in1_valid  = 1;
    in1_data   = 32'hB1B1;
    #1;

    if (out0_valid !== 1)
      $fatal(1, "FAIL: out0_valid should be 1 with diagonal route");
    if (out0_data !== 32'hA0A0)
      $fatal(1, "FAIL: out0_data mismatch: expected A0A0 got %08h", out0_data);
    if (out1_valid !== 1)
      $fatal(1, "FAIL: out1_valid should be 1 with diagonal route");
    if (out1_data !== 32'hB1B1)
      $fatal(1, "FAIL: out1_data mismatch: expected B1B1 got %08h", out1_data);

    in0_valid = 0;
    in1_valid = 0;

    $display("PASS: tb_switch_top");
    $finish;
  end

  initial begin
    #100000;
    $fatal(1, "FAIL: watchdog timeout");
  end

endmodule
