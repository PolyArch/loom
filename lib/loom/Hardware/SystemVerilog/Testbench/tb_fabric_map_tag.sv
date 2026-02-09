//===-- tb_fabric_map_tag.sv - Parameterized map_tag test ------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_fabric_map_tag;
  parameter int DATA_WIDTH    = 32;
  parameter int IN_TAG_WIDTH  = 4;
  parameter int OUT_TAG_WIDTH = 4;
  parameter int TABLE_SIZE    = 4;

  localparam int ENTRY_WIDTH  = 1 + IN_TAG_WIDTH + OUT_TAG_WIDTH;
  localparam int IN_PW        = DATA_WIDTH + IN_TAG_WIDTH;
  localparam int OUT_PW       = DATA_WIDTH + OUT_TAG_WIDTH;
  localparam int CONFIG_WIDTH = TABLE_SIZE * ENTRY_WIDTH;

  logic clk, rst_n;
  logic in_valid, in_ready;
  logic [IN_PW-1:0] in_data;
  logic out_valid, out_ready;
  logic [OUT_PW-1:0] out_data;
  logic [CONFIG_WIDTH > 0 ? CONFIG_WIDTH-1 : 0 : 0] cfg_data;
  logic        error_valid;
  logic [15:0] error_code;

  fabric_map_tag #(
    .DATA_WIDTH(DATA_WIDTH),
    .IN_TAG_WIDTH(IN_TAG_WIDTH),
    .OUT_TAG_WIDTH(OUT_TAG_WIDTH),
    .TABLE_SIZE(TABLE_SIZE)
  ) dut (.*);

  initial begin : clk_gen
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin : test
    integer pass_count;
    pass_count = 0;
    rst_n = 0;
    in_valid = 0;
    out_ready = 1;
    in_data = '0;
    cfg_data = '0;

    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // Check 1: no error after reset
    if (error_valid !== 1'b0) begin : check_reset
      $fatal(1, "error_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    // Check 2: valid mapping - no error
    // Entry 0: valid=1, src_tag=1, dst_tag=2
    cfg_data = '0;
    cfg_data[ENTRY_WIDTH-1] = 1'b1;
    cfg_data[OUT_TAG_WIDTH +: IN_TAG_WIDTH] = IN_TAG_WIDTH'(1);
    cfg_data[0 +: OUT_TAG_WIDTH] = OUT_TAG_WIDTH'(2);
    // Send data with tag=1
    in_data = '0;
    in_data[DATA_WIDTH +: IN_TAG_WIDTH] = IN_TAG_WIDTH'(1);
    in_data[DATA_WIDTH-1:0] = DATA_WIDTH'(42);
    in_valid = 1;
    @(posedge clk);
    if (error_valid !== 1'b0) begin : check_valid_map
      $fatal(1, "error after valid mapping");
    end
    pass_count = pass_count + 1;
    in_valid = 0;
    @(posedge clk);

    // Check 3: RT_MAP_TAG_NO_MATCH - send tag not in table
    // Reset to clear any latched errors
    rst_n = 0;
    repeat (2) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    // cfg_data still has entry 0 with src_tag=1; send tag=5 (no match)
    in_data = '0;
    in_data[DATA_WIDTH +: IN_TAG_WIDTH] = IN_TAG_WIDTH'(5);
    in_valid = 1;
    @(posedge clk);
    @(posedge clk);
    if (error_valid !== 1'b1) begin : check_no_match
      $fatal(1, "expected RT_MAP_TAG_NO_MATCH error");
    end
    if (error_code !== RT_MAP_TAG_NO_MATCH) begin : check_no_match_code
      $fatal(1, "wrong error code for no match: got %0d", error_code);
    end
    pass_count = pass_count + 1;
    in_valid = 0;

    // Check 4: CFG_MAP_TAG_DUP_TAG - duplicate source tags
    rst_n = 0;
    repeat (2) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
    // Set entry 0 and entry 1 both valid with same src_tag=3
    cfg_data = '0;
    // Entry 0: valid=1, src_tag=3, dst_tag=0
    cfg_data[ENTRY_WIDTH-1] = 1'b1;
    cfg_data[OUT_TAG_WIDTH +: IN_TAG_WIDTH] = IN_TAG_WIDTH'(3);
    // Entry 1: valid=1, src_tag=3, dst_tag=1
    cfg_data[ENTRY_WIDTH + ENTRY_WIDTH-1] = 1'b1;
    cfg_data[ENTRY_WIDTH + OUT_TAG_WIDTH +: IN_TAG_WIDTH] = IN_TAG_WIDTH'(3);
    cfg_data[ENTRY_WIDTH +: OUT_TAG_WIDTH] = OUT_TAG_WIDTH'(1);
    @(posedge clk);
    @(posedge clk);
    if (error_valid !== 1'b1) begin : check_dup_tag
      $fatal(1, "expected CFG_MAP_TAG_DUP_TAG error");
    end
    if (error_code !== CFG_MAP_TAG_DUP_TAG) begin : check_dup_code
      $fatal(1, "wrong error code for dup tag: got %0d", error_code);
    end
    pass_count = pass_count + 1;

    $display("PASS: tb_fabric_map_tag DW=%0d ITW=%0d OTW=%0d TS=%0d (%0d checks)",
             DATA_WIDTH, IN_TAG_WIDTH, OUT_TAG_WIDTH, TABLE_SIZE, pass_count);
    $finish;
  end

  initial begin : timeout
    #10000;
    $fatal(1, "TIMEOUT");
  end
endmodule
