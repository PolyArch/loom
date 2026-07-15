// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// op_list cannot be empty.
fabric.module @op_empty_list(%a : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> () {
      // expected-error @+1 {{'op_list' must be non-empty}}
      %0 = fabric.op [] (%fa) : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// Unknown op symbol.
fabric.module @op_unknown_symbol(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> () {
      // expected-error @+1 {{is not a fabric.op-supported software op}}
      %0 = fabric.op [@arith.no_such_op] (%fa, %fb)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// arith.constant is explicitly disallowed (constants must come from
// fabric.op[@dataflow.constant]).
fabric.module @op_rejects_arith_constant(%a : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> () {
      // expected-error @+1 {{is not a fabric.op-supported software op}}
      %0 = fabric.op [@arith.constant] (%fa)
           : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// Two singleton ops cannot share a fabric.op.
fabric.module @op_two_singletons(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> () {
      // expected-error @+1 {{is not in any multi-member hardware-share group}}
      %0 = fabric.op [@arith.muli, @arith.addf] (%fa, %fb)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// Two ops from different groups cannot share a fabric.op.
fabric.module @op_different_groups(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> () {
      // expected-error @+1 {{do not belong to the same hardware-share group}}
      %0 = fabric.op [@arith.addi, @arith.divsi] (%fa, %fb)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// Multi-op programmed without op_sel.
fabric.module @op_missing_op_sel(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> () {
      // expected-error @+1 {{'sw_configs' must contain key 'op_sel'}}
      %0 = fabric.op [@arith.addi, @arith.subi] (%fa, %fb)
           {sw_configs = {something = "else"}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// op_sel value not in op_list.
fabric.module @op_bad_op_sel(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> () {
      // expected-error @+1 {{'sw_configs.op_sel' value "arith.muli" is not one of the symbols listed in 'op_list'}}
      %0 = fabric.op [@arith.addi, @arith.subi] (%fa, %fb)
           {sw_configs = {op_sel = "arith.muli"}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// Mismatched port count.
fabric.module @op_bad_port_count(%a : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>) -> () {
      // expected-error @+1 {{port count (1->1) does not match the supported software ops (2->1)}}
      %0 = fabric.op [@arith.addi] (%fa)
           : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// Wrong fixed-port width: dataflow.stream's rwc port must be bits<1> not bits<0>.
fabric.module @op_stream_bad_rwc(%lb : !fabric.bits<32>, %ub : !fabric.bits<32>, %step : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %lb : !fabric.bits<32>,
                    %pb = %ub : !fabric.bits<32>,
                    %pc = %step : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>,
              %fc = %pc : !fabric.bits<32>) -> () {
      // expected-error @+1 {{output port #1 has width 0 but software op(s) require width 1}}
      %i, %r = fabric.op [@dataflow.stream] (%fa, %fb, %fc)
               : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
                 -> (!fabric.bits<32>, !fabric.bits<0>)
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// dataflow.constant must have bits<0> input (none-typed ctrl).
fabric.module @op_constant_bad_ctrl(%ctrl : !fabric.bits<1>) {
  fabric.pe [spatial] (%pa = %ctrl : !fabric.bits<1>) -> !fabric.bits<1> {
    fabric.fu(%fa = %pa : !fabric.bits<1>) -> () {
      // expected-error @+1 {{input port #0 has width 1 but software op(s) require width 0}}
      %0 = fabric.op [@dataflow.constant] (%fa)
           {sw_configs = {const_hex_value = "0x1"}}
           : (!fabric.bits<1>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// hw_params must be a length-1 array.
fabric.module @op_bad_hw_params(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> () {
      // expected-error @+1 {{'hw_params' must be a length-1 array}}
      %0 = fabric.op [@arith.muli] (%fa, %fb)
           {hw_params = [{}, {}]}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// Inconsistent type-parameter widths: dataflow.stream's three input ports must
// all share one width. The FU is at bits<32> uniformly; the inconsistent
// bits<64> input is materialized internally via a width-changing op.
fabric.module @op_stream_inconsistent_t(%lb : !fabric.bits<32>, %ub : !fabric.bits<32>, %step : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %lb : !fabric.bits<32>,
                    %pb = %ub : !fabric.bits<32>,
                    %pc = %step : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>,
              %fc = %pc : !fabric.bits<32>) -> () {
      %fb64 = fabric.op [@arith.sitofp] (%fb)
              : (!fabric.bits<32>) -> !fabric.bits<64>
      // expected-error @+1 {{requires the same width on all ports tied to its type parameter}}
      %i, %r = fabric.op [@dataflow.stream] (%fa, %fb64, %fc)
               : (!fabric.bits<32>, !fabric.bits<64>, !fabric.bits<32>)
                 -> (!fabric.bits<32>, !fabric.bits<1>)
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// dataflow.sync: in/out counts must match.
fabric.module @op_sync_unequal_counts(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> () {
      // expected-error @+1 {{@dataflow.sync requires equal input/output counts}}
      %0 = fabric.op [@dataflow.sync] (%fa, %fb)
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// dataflow.sync: bitmask length must equal port count.
fabric.module @op_sync_bad_bitmask_len(%a : !fabric.bits<32>, %b : !fabric.bits<32>, %c : !fabric.bits<32>, %d : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>,
                    %pc = %c : !fabric.bits<32>,
                    %pd = %d : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>,
              %fc = %pc : !fabric.bits<32>,
              %fd = %pd : !fabric.bits<32>) -> () {
      // expected-error @+1 {{'sw_configs.bitmask' length (3) must equal port count (4)}}
      %w, %x, %y, %z = fabric.op [@dataflow.sync] (%fa, %fb, %fc, %fd)
                       {sw_configs = {bitmask = "101"}}
                       : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
                         -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// dataflow.sync: bitmask must contain only '0' / '1'.
fabric.module @op_sync_bad_bitmask_chars(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> () {
      // expected-error @+1 {{'sw_configs.bitmask' must contain only '0' and '1'}}
      %x, %y = fabric.op [@dataflow.sync] (%fa, %fb)
               {sw_configs = {bitmask = "1x"}}
               : (!fabric.bits<32>, !fabric.bits<32>)
                 -> (!fabric.bits<32>, !fabric.bits<32>)
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// dataflow.mux with 2 data inputs requires bits<1> sel. The faulty bits<32>
// sel is taken directly from a PE/FU input at bits<32> (PE width matches
// data width).
fabric.module @op_mux2_bad_sel(%sel : !fabric.bits<32>, %a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%psel = %sel : !fabric.bits<32>,
                    %pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fsel = %psel : !fabric.bits<32>,
              %fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> () {
      // expected-error @+1 {{sel port (input #0) width 32 must be 1}}
      %0 = fabric.op [@dataflow.mux] (%fsel, %fa, %fb)
           : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// dataflow.mux with >2 data inputs requires sel width = index width
// (default 32). Faulty bits<1> sel is generated internally via cmpi.
fabric.module @op_mux3_bad_sel(%a : !fabric.bits<16>, %b : !fabric.bits<16>, %c : !fabric.bits<16>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<16>,
                    %pb = %b : !fabric.bits<16>,
                    %pc = %c : !fabric.bits<16>) -> !fabric.bits<16> {
    fabric.fu(%fa = %pa : !fabric.bits<16>,
              %fb = %pb : !fabric.bits<16>,
              %fc = %pc : !fabric.bits<16>) -> () {
      %sel = fabric.op [@arith.cmpi] (%fa, %fb)
             : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<1>
      // expected-error @+1 {{sel port (input #0) width 1 must be 32}}
      %0 = fabric.op [@dataflow.mux] (%sel, %fa, %fb, %fc)
           : (!fabric.bits<1>, !fabric.bits<16>, !fabric.bits<16>, !fabric.bits<16>)
             -> !fabric.bits<16>
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// dataflow.demux with 2 outs requires bits<1> sel.
fabric.module @op_demux2_bad_sel(%sel : !fabric.bits<8>, %in : !fabric.bits<8>) {
  fabric.pe [spatial] (%psel = %sel : !fabric.bits<8>,
                    %pa = %in : !fabric.bits<8>) -> !fabric.bits<8> {
    fabric.fu(%fsel_8 = %psel : !fabric.bits<8>,
              %fin = %pa : !fabric.bits<8>) -> () {
      %fsel = fabric.op [@arith.sitofp] (%fsel_8)
              : (!fabric.bits<8>) -> !fabric.bits<32>
      // expected-error @+1 {{sel port (input #0) width 32 must be 1}}
      %a, %b = fabric.op [@dataflow.demux] (%fsel, %fin)
               : (!fabric.bits<32>, !fabric.bits<8>)
                 -> (!fabric.bits<8>, !fabric.bits<8>)
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// dataflow.mux: data inputs must match output width. The PE/FU is at the
// output width (bits<16>); the bits<1> sel and bits<32> mismatched data
// input are both materialized internally.
fabric.module @op_mux_data_mismatch(%a : !fabric.bits<16>, %b : !fabric.bits<16>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<16>,
                    %pb = %b : !fabric.bits<16>) -> !fabric.bits<16> {
    fabric.fu(%fa = %pa : !fabric.bits<16>,
              %fb = %pb : !fabric.bits<16>) -> () {
      %sel = fabric.op [@arith.cmpi] (%fa, %fb)
             : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<1>
      %fb32 = fabric.op [@arith.sitofp] (%fb)
              : (!fabric.bits<16>) -> !fabric.bits<32>
      // expected-error @+1 {{@dataflow.mux input #2 width 32 must match output width 16}}
      %0 = fabric.op [@dataflow.mux] (%sel, %fa, %fb32)
           : (!fabric.bits<1>, !fabric.bits<16>, !fabric.bits<32>) -> !fabric.bits<16>
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// hw_params allowed-set check: sw_configs value not in hw_params allowed array.
fabric.module @op_sw_value_not_in_hw_set(%lb : !fabric.bits<32>, %ub : !fabric.bits<32>, %step : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %lb : !fabric.bits<32>,
                    %pb = %ub : !fabric.bits<32>,
                    %pc = %step : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>,
              %fc = %pc : !fabric.bits<32>) -> () {
      // expected-error @+1 {{'sw_configs["step_op"]' value "%=" is not in the 'hw_params["step_op"]' allowed set}}
      %i, %r = fabric.op [@dataflow.stream] (%fa, %fb, %fc)
               {hw_params = [{step_op = ["+=", "/="], cont_cond = ["<", ">"]}],
                sw_configs = {step_op = "%=", cont_cond = "<"}}
               : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
                 -> (!fabric.bits<32>, !fabric.bits<1>)
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// hw_params allowed-set check: hw value for shared key must be ArrayAttr.
fabric.module @op_hw_value_not_array(%lb : !fabric.bits<32>, %ub : !fabric.bits<32>, %step : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %lb : !fabric.bits<32>,
                    %pb = %ub : !fabric.bits<32>,
                    %pc = %step : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>,
              %fc = %pc : !fabric.bits<32>) -> () {
      // expected-error @+1 {{'hw_params["step_op"]' must be an array of allowed values}}
      %i, %r = fabric.op [@dataflow.stream] (%fa, %fb, %fc)
               {hw_params = [{step_op = "+="}],
                sw_configs = {step_op = "+=", cont_cond = "<"}}
               : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
                 -> (!fabric.bits<32>, !fabric.bits<1>)
      fabric.yield
    }
  }
  fabric.yield
}

// -----
// Normalized hardware modes are selected only by mode index.
fabric.module @op_normalized_mode_out_of_range(%a : !fabric.bits<32>,
                                               %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                      %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> () {
      // expected-error @+1 {{'sw_configs.mode' is out of range for hw_params}}
      %v = fabric.op [@arith.addi] (%fa, %fb)
           {hw_params = [
             {op = @arith.addi, function_type = (i32, i32) -> i32,
              input_ports = [0 : i32, 1 : i32],
              output_ports = [0 : i32], attributes = {}}
           ], sw_configs = {mode = 1 : i32}}
           : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield
    }
  }
  fabric.yield
}
