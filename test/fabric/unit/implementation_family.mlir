// RUN: loom --verify-diagnostics %s

builtin.module attributes {
  fabric.implementation_family =
      #fabric.implementation_family<ScalarIntegerAddSub>
} {
}

fabric.module @typed_family(%lhs: !fabric.bits<32>,
                            %rhs: !fabric.bits<32>) {
  fabric.pe [spatial] (%pe_lhs = %lhs : !fabric.bits<32>,
                       %pe_rhs = %rhs : !fabric.bits<32>)
      -> !fabric.bits<32> {
    %result = fabric.fu(%fu_lhs = %pe_lhs : !fabric.bits<32>,
                        %fu_rhs = %pe_rhs : !fabric.bits<32>)
        -> !fabric.bits<32> {
      %sum = fabric.op [@arith.addi, @arith.subi] (%fu_lhs, %fu_rhs) {
        implementation_family =
            #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %sum : !fabric.bits<32>
    }
  }
  fabric.yield
}

fabric.module @wrong_member(%lhs: !fabric.bits<32>,
                            %rhs: !fabric.bits<32>) {
  fabric.pe [spatial] (%pe_lhs = %lhs : !fabric.bits<32>,
                       %pe_rhs = %rhs : !fabric.bits<32>)
      -> !fabric.bits<32> {
    %result = fabric.fu(%fu_lhs = %pe_lhs : !fabric.bits<32>,
                        %fu_rhs = %pe_rhs : !fabric.bits<32>)
        -> !fabric.bits<32> {
      // expected-error @+1 {{op_list member @arith.muli is not admitted by implementation family ScalarIntegerAddSub}}
      %product = fabric.op [@arith.muli] (%fu_lhs, %fu_rhs) {
        implementation_family =
            #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %product : !fabric.bits<32>
    }
  }
  fabric.yield
}

fabric.module @missing_family(%lhs: !fabric.bits<32>,
                              %rhs: !fabric.bits<32>) {
  fabric.pe [spatial] (%pe_lhs = %lhs : !fabric.bits<32>,
                       %pe_rhs = %rhs : !fabric.bits<32>)
      -> !fabric.bits<32> {
    %result = fabric.fu(%fu_lhs = %pe_lhs : !fabric.bits<32>,
                        %fu_rhs = %pe_rhs : !fabric.bits<32>)
        -> !fabric.bits<32> {
      // expected-error @+1 {{requires an explicit implementation_family}}
      %sum = fabric.op [@arith.addi] (%fu_lhs, %fu_rhs) {
        hw_params = {integer_widths = [1 : i32]}
      }
          : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %sum : !fabric.bits<32>
    }
  }
  fabric.yield
}
