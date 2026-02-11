//===-- tb_memory_top.sv - E2E smoke test for memory ----------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

`include "fabric_common.svh"

module tb_memory_top;
  logic clk, rst_n;

  // LoadPE inputs
  logic ld_addr_valid, ld_addr_ready;
  logic [63:0] ld_addr_data;
  logic ld_data_in_valid, ld_data_in_ready;
  logic [31:0] ld_data_in_data;
  logic ld_ctrl_valid, ld_ctrl_ready;
  logic ld_ctrl_data;

  // StorePE inputs
  logic st_addr_valid, st_addr_ready;
  logic [63:0] st_addr_data;
  logic st_data_valid, st_data_ready;
  logic [31:0] st_data_data;
  logic st_ctrl_valid, st_ctrl_ready;
  logic st_ctrl_data;

  // LoadPE data output
  logic ld_data_out_valid, ld_data_out_ready;
  logic [31:0] ld_data_out_data;

  // Memory outputs
  logic lddata_valid, lddata_ready;
  logic [31:0] lddata_data;
  logic lddone_valid, lddone_ready;
  logic lddone_data;
  logic stdone_valid, stdone_ready;
  logic stdone_data;

  // Config ports
  logic [3:0] ld0_cfg_data;
  logic [3:0] st0_cfg_data;

  logic        error_valid;
  logic [15:0] error_code;

  memory_top dut (.*);

  initial begin : clk_gen
    clk = 0;
    forever #5 clk = ~clk;
  end

`ifdef DUMP_FST
  initial begin : dump_fst
    $dumpfile("waves.fst");
    $dumpvars(0, tb_memory_top);
  end
`endif
`ifdef DUMP_FSDB
  initial begin : dump_fsdb
    $fsdbDumpfile("waves.fsdb");
    $fsdbDumpvars(0, tb_memory_top, "+mda");
  end
`endif

  initial begin : test
    integer pass_count;
    integer iter_var0;
    pass_count = 0;
    rst_n = 0;
    ld_addr_valid = 0;
    ld_data_in_valid = 0;
    ld_ctrl_valid = 0;
    st_addr_valid = 0;
    st_data_valid = 0;
    st_ctrl_valid = 0;
    ld_data_out_ready = 1;
    lddata_ready = 1;
    lddone_ready = 1;
    stdone_ready = 1;
    ld_addr_data = '0;
    ld_data_in_data = '0;
    ld_ctrl_data = '0;
    st_addr_data = '0;
    st_data_data = '0;
    st_ctrl_data = '0;
    ld0_cfg_data = '0;
    st0_cfg_data = '0;

    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // Check 1: no error after reset
    if (error_valid !== 1'b0) begin : check_reset
      $fatal(1, "error_valid should be 0 after reset");
    end
    pass_count = pass_count + 1;

    // Check 2: store via StorePE then load via LoadPE
    // StorePE: addr=5, data=0xBEEF, ctrl=1 (fire)
    st_addr_data = 64'd5;
    st_addr_valid = 1;
    st_data_data = 32'hBEEF;
    st_data_valid = 1;
    st_ctrl_data = 1'b1;
    st_ctrl_valid = 1;
    @(posedge clk);
    st_addr_valid = 0;
    st_data_valid = 0;
    st_ctrl_valid = 0;

    // Wait for stdone
    iter_var0 = 0;
    while (iter_var0 < 20 && !stdone_valid) begin : wait_stdone
      $display("[%0t] wait_stdone cycle %0d: stdone_valid=%0b", $time, iter_var0, stdone_valid);
      $display("  st_addr_ready=%0b st_data_ready=%0b st_ctrl_ready=%0b",
               st_addr_ready, st_data_ready, st_ctrl_ready);
      $display("  error_valid=%0b error_code=0x%04h", error_valid, error_code);
      @(posedge clk);
      iter_var0 = iter_var0 + 1;
    end
    if (!stdone_valid) begin : check_stdone
      $display("[%0t] FINAL: stdone_valid=%0b after %0d cycles", $time, stdone_valid, iter_var0);
      $display("  st_addr_ready=%0b st_data_ready=%0b st_ctrl_ready=%0b",
               st_addr_ready, st_data_ready, st_ctrl_ready);
      $fatal(1, "stdone_valid not asserted within 20 cycles");
    end
    @(posedge clk);

    // LoadPE: addr=5, ctrl=1 (fire)
    ld_addr_data = 64'd5;
    ld_addr_valid = 1;
    ld_data_in_data = '0;
    ld_data_in_valid = 1;
    ld_ctrl_data = 1'b1;
    ld_ctrl_valid = 1;
    @(posedge clk);
    ld_addr_valid = 0;
    ld_data_in_valid = 0;
    ld_ctrl_valid = 0;

    // Wait for lddata and verify round-trip data integrity
    iter_var0 = 0;
    while (iter_var0 < 20 && !lddata_valid) begin : wait_lddata
      @(posedge clk);
      iter_var0 = iter_var0 + 1;
    end
    if (!lddata_valid) begin : check_lddata_seen
      $fatal(1, "lddata_valid not asserted within 20 cycles");
    end
    if (lddata_data !== 32'hBEEF) begin : check_ld_data
      $fatal(1, "load data mismatch: expected 0xBEEF, got 0x%0h", lddata_data);
    end
    pass_count = pass_count + 1;

    // Check 3: no error after store/load
    if (error_valid !== 1'b0) begin : check_no_err
      $fatal(1, "unexpected error after store/load: code=%0d", error_code);
    end
    pass_count = pass_count + 1;

    $display("PASS: memory_top (%0d checks)", pass_count);
    $finish;
  end

  initial begin : timeout
    #10000;
    $fatal(1, "TIMEOUT");
  end
endmodule
