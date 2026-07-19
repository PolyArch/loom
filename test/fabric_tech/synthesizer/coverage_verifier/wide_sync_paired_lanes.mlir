// RUN: loom-coverage-test %s | FileCheck %s --check-prefix=COVERAGE
// RUN: loom %s | FileCheck %s --check-prefix=FABRIC

fabric.module @wide_sync(
    %a: !fabric.bits<64>, %b: !fabric.bits<64>,
    %c: !fabric.bits<64>, %d: !fabric.bits<64>) {
  fabric.pe [spatial] (
      %pa = %a : !fabric.bits<64>, %pb = %b : !fabric.bits<64>,
      %pc = %c : !fabric.bits<64>, %pd = %d : !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>,
          !fabric.bits<64>, !fabric.bits<64>) {
    %r:4 = fabric.fu(
        %x0 = %pa : !fabric.bits<64>, %x1 = %pb : !fabric.bits<64>,
        %x2 = %pc : !fabric.bits<64>, %x3 = %pd : !fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>,
            !fabric.bits<64>, !fabric.bits<64>)
        attributes {valid_encodings = [{
          outputs = [0 : i32, 1 : i32, 2 : i32, 3 : i32],
          resources = [{resource = 0 : i32, mode = 0 : i32}]
        }]} {
      %y0, %y1, %y2, %y3 = fabric.op [@dataflow.sync]
          (%x0, %x1, %x2, %x3) {
            hw_params = [{
              op = @dataflow.sync,
              function_type = (i32, i64, i8, i16) ->
                              (i32, i64, i8, i16),
              input_ports = [2 : i32, 0 : i32, 3 : i32, 1 : i32],
              output_ports = [3 : i32, 1 : i32, 2 : i32, 0 : i32],
              attributes = {}
            }],
            paired_lanes = [
              {input_port = 2 : i32, output_port = 3 : i32,
               mask_bit = 1 : i32},
              {input_port = 0 : i32, output_port = 1 : i32,
               mask_bit = 3 : i32},
              {input_port = 3 : i32, output_port = 2 : i32,
               mask_bit = 0 : i32},
              {input_port = 1 : i32, output_port = 0 : i32,
               mask_bit = 2 : i32}
            ]
          } : (!fabric.bits<64>, !fabric.bits<64>,
               !fabric.bits<64>, !fabric.bits<64>)
              -> (!fabric.bits<64>, !fabric.bits<64>,
                  !fabric.bits<64>, !fabric.bits<64>)
      fabric.yield %y0, %y1, %y2, %y3 :
          !fabric.bits<64>, !fabric.bits<64>,
          !fabric.bits<64>, !fabric.bits<64>
    }
  }
  fabric.yield
}

func.func @selected_non_prefix(%a: i16, %b: i32) -> i16
    attributes {loom.coverage_input = true} {
  %x, %y = dataflow.sync %a, %b : (i16, i32) -> (i16, i32)
  return %x : i16
}

func.func @selected_full_width(
    %a: i32, %b: i64, %c: i8, %d: i16) -> i32
    attributes {loom.coverage_input = true} {
  %x0, %x1, %x2, %x3 = dataflow.sync %a, %b, %c, %d :
      (i32, i64, i8, i16) -> (i32, i64, i8, i16)
  return %x0 : i32
}

func.func @duplicate_physical_lane(%a: i32, %b: i32) -> i32
    attributes {loom.coverage_input = true} {
  %x, %y = dataflow.sync %a, %b : (i32, i32) -> (i32, i32)
  return %x : i32
}

func.func @incompatible_lane_type(%a: f32) -> f32
    attributes {loom.coverage_input = true} {
  %x = dataflow.sync %a : (f32) -> (f32)
  return %x : f32
}

// COVERAGE: coverage[0] funcname=selected_non_prefix matched=true index=0
// COVERAGE-SAME: lanes=[0:{1->0,2->3}] bitmasks=[0:0110]
// COVERAGE-NEXT: coverage[1] funcname=selected_full_width matched=true index=0
// COVERAGE-SAME: lanes=[0:{2->3,0->1,3->2,1->0}] bitmasks=[0:1111]
// COVERAGE-NEXT: coverage[2] funcname=duplicate_physical_lane matched=false index=none
// COVERAGE-NEXT: coverage[3] funcname=incompatible_lane_type matched=false index=none
// COVERAGE-NEXT: all_covered=false

// FABRIC-COUNT-1: valid_encodings = [
// FABRIC-COUNT-1: fabric.op [@dataflow.sync]
// FABRIC-SAME: paired_lanes = [
// FABRIC-NOT: sw_configs
