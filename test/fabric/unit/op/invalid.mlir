// RUN: loom %s -split-input-file -verify-diagnostics

fabric.module @empty_list(%a : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> () {
      // expected-error @+1 {{'op_list' must be non-empty}}
      %value = fabric.op [] (%fa) {
        implementation_family =
            #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// -----

fabric.module @unknown_schema(%a : !fabric.bits<32>,
                              %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> () {
      // expected-error @+1 {{op_list member @arith.no_such_op is not a registered canonical operation schema}}
      %value = fabric.op [@arith.no_such_op] (%fa, %fb) {
        implementation_family =
            #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// -----

fabric.module @wrong_family(%a : !fabric.bits<32>,
                            %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> () {
      // expected-error @+1 {{op_list member @arith.muli is not admitted by implementation family ScalarIntegerAddSub}}
      %value = fabric.op [@arith.muli] (%fa, %fb) {
        implementation_family =
            #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// -----

fabric.module @duplicate_member(%a : !fabric.bits<32>,
                                %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> () {
      // expected-error @+1 {{op_list contains duplicate member @arith.addi}}
      %value = fabric.op [@arith.addi, @arith.addi] (%fa, %fb) {
        implementation_family =
            #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}
