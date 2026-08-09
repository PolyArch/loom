module chipware_scalar_integer_multiply_testbench;
  logic [7:0] data_input_0;
  logic [7:0] data_input_1;
  logic [7:0] data_output_0;

  chipware_scalar_integer_multiply dut (
    .data_input_0(data_input_0),
    .data_input_1(data_input_1),
    .data_output_0(data_output_0)
  );

  task automatic check_product(input logic [7:0] lhs,
                               input logic [7:0] rhs);
    logic [7:0] expected;
    begin
      data_input_0 = lhs;
      data_input_1 = rhs;
      expected = lhs * rhs;
      #1;
      if (data_output_0 !== expected)
        $fatal(1, "product mismatch: %0h * %0h = %0h, got %0h", lhs, rhs,
               expected, data_output_0);
    end
  endtask

  initial begin
    for (int unsigned lhs = 0; lhs < 256; ++lhs)
      for (int unsigned rhs = 0; rhs < 256; ++rhs)
        check_product(lhs[7:0], rhs[7:0]);
    $display("LOOM_CHIPWARE_XCELIUM_PASS vectors=65536");
    $finish;
  end
endmodule
