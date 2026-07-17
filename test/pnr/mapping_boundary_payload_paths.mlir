// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph boundary_payload --hardware-mlir %s --hardware boundary_data_adg --workload boundary_data --output %t.data.csv --artifact %t.data.json
// RUN: FileCheck %s --check-prefix=DATA-CSV < %t.data.csv
// RUN: FileCheck %s --check-prefix=DATA-JSON < %t.data.json
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph boundary_payload --hardware-mlir %s --hardware boundary_tag_adg --workload boundary_tag --output %t.tag.csv --artifact %t.tag.json
// RUN: FileCheck %s --check-prefix=TAG-CSV < %t.tag.csv
// RUN: FileCheck %s --check-prefix=TAG-JSON < %t.tag.json
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph boundary_payload --hardware-mlir %s --hardware boundary_t2t_adg --workload boundary_t2t --output %t.t2t.csv --artifact %t.t2t.json
// RUN: FileCheck %s --check-prefix=T2T-CSV < %t.t2t.csv
// RUN: FileCheck %s --check-prefix=T2T-JSON < %t.t2t.json
// RUN: loom-pnr-map --dfg-mlir %t.lowered.mlir --graph boundary_payload --hardware-mlir %s --hardware boundary_t2s_tag_adg --workload boundary_t2s_tag --output %t.t2s-tag.csv --artifact %t.t2s-tag.json
// RUN: FileCheck %s --check-prefix=T2S-TAG-CSV < %t.t2s-tag.csv
// RUN: FileCheck %s --check-prefix=T2S-TAG-JSON < %t.t2s-tag.json

// DATA-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// DATA-CSV-NEXT: boundary_data,boundary_data_adg,boundary_data__boundary_payload__boundary_data_adg,3,2,0,0,pass,mapped software graph to fabric resources
// DATA-JSON-DAG: "status": "pass"
// DATA-JSON-DAG: "segment_kind": "boundary_crossing"

// TAG-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// TAG-CSV-NEXT: boundary_tag,boundary_tag_adg,boundary_tag__boundary_payload__boundary_tag_adg,3,1,1,0,fail,unrouted software edges lack Fabric ADG connectivity
// TAG-JSON-DAG: "status": "fail"
// TAG-JSON-DAG: "routed_edges": 1
// TAG-JSON-DAG: "unrouted_edges": 1

// T2T-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// T2T-CSV-NEXT: boundary_t2t,boundary_t2t_adg,boundary_t2t__boundary_payload__boundary_t2t_adg,3,2,0,0,pass,mapped software graph to fabric resources
// T2T-JSON-DAG: "status": "pass"
// T2T-JSON-DAG: "segment_kind": "boundary_crossing"

// T2S-TAG-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// T2S-TAG-CSV-NEXT: boundary_t2s_tag,boundary_t2s_tag_adg,boundary_t2s_tag__boundary_payload__boundary_t2s_tag_adg,3,1,1,0,fail,unrouted software edges lack Fabric ADG connectivity
// T2S-TAG-JSON-DAG: "status": "fail"
// T2S-TAG-JSON-DAG: "routed_edges": 1
// T2S-TAG-JSON-DAG: "unrouted_edges": 1

module {
  dataflow.graph.func private @boundary_payload(
      %ctrl: none, %lhs: i32, %rhs: i32) -> (none, i32) {
    %sum = arith.addi %lhs, %rhs : i32
    %product = arith.muli %sum, %rhs : i32
    dataflow.graph.return %ctrl, %product : none, i32
  }

  fabric.module @boundary_data_adg(%ctrl : !fabric.bits<0>,
                                   %lhs : !fabric.bits<32>,
                                   %rhs : !fabric.bits<32>,
                                   %tag : !fabric.bits<32>) {
    %rhs_to_add, %rhs_to_mul = fabric.switch [spatial] %rhs
        [{connectivity_table = ["1", "1"]}]
        : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
    %sum = fabric.pe [spatial] (%a = %lhs : !fabric.bits<32>,
                               %b = %rhs_to_add : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%fa = %a : !fabric.bits<32>,
                %fb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
        %value = fabric.op [@arith.addi] (%fa, %fb)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %value : !fabric.bits<32>
      }
    }
    %tagged = fabric.boundary [s2t] %sum, %tag
        : (!fabric.bits<32>, !fabric.bits<32>)
       -> !fabric.bits_tag<32, 32>
    %untagged = fabric.boundary [t2s] %tagged
        : !fabric.bits_tag<32, 32> -> !fabric.bits<32>
    fabric.pe [spatial] (%a = %untagged : !fabric.bits<32>,
                         %b = %rhs_to_mul : !fabric.bits<32>,
                         %pc = %ctrl : !fabric.bits<0> to !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%fa = %a : !fabric.bits<32>,
                %fb = %b : !fabric.bits<32>,
                %token = %pc : !fabric.bits<32> to !fabric.bits<0>)
          -> !fabric.bits<32> {
        %value = fabric.op [@arith.muli] (%fa, %fb)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        %done, %published = fabric.op [@dataflow.sync] (%token, %value)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<32>)
              -> (!fabric.bits<0>, !fabric.bits<32>)
        fabric.yield %value : !fabric.bits<32>
      }
    }
    fabric.yield
  }

  fabric.module @boundary_tag_adg(%ctrl : !fabric.bits<0>,
                                  %lhs : !fabric.bits<32>,
                                  %rhs : !fabric.bits<32>,
                                  %data : !fabric.bits<32>) {
    %rhs_to_add, %rhs_to_mul = fabric.switch [spatial] %rhs
        [{connectivity_table = ["1", "1"]}]
        : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
    %sum = fabric.pe [spatial] (%a = %lhs : !fabric.bits<32>,
                               %b = %rhs_to_add : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%fa = %a : !fabric.bits<32>,
                %fb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
        %value = fabric.op [@arith.addi] (%fa, %fb)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %value : !fabric.bits<32>
      }
    }
    %tagged = fabric.boundary [s2t] %data, %sum
        : (!fabric.bits<32>, !fabric.bits<32>)
       -> !fabric.bits_tag<32, 32>
    %untagged = fabric.boundary [t2s] %tagged
        : !fabric.bits_tag<32, 32> -> !fabric.bits<32>
    fabric.pe [spatial] (%a = %untagged : !fabric.bits<32>,
                         %b = %rhs_to_mul : !fabric.bits<32>,
                         %pc = %ctrl : !fabric.bits<0> to !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%fa = %a : !fabric.bits<32>,
                %fb = %b : !fabric.bits<32>,
                %token = %pc : !fabric.bits<32> to !fabric.bits<0>)
          -> !fabric.bits<32> {
        %value = fabric.op [@arith.muli] (%fa, %fb)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        %done, %published = fabric.op [@dataflow.sync] (%token, %value)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<32>)
              -> (!fabric.bits<0>, !fabric.bits<32>)
        fabric.yield %value : !fabric.bits<32>
      }
    }
    fabric.yield
  }

  fabric.module @boundary_t2t_adg(%ctrl : !fabric.bits<0>,
                                  %lhs : !fabric.bits<32>,
                                  %rhs : !fabric.bits<32>,
                                  %tag : !fabric.bits<4>) {
    %rhs_to_add, %rhs_to_mul = fabric.switch [spatial] %rhs
        [{connectivity_table = ["1", "1"]}]
        : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
    %sum = fabric.pe [spatial] (%a = %lhs : !fabric.bits<32>,
                               %b = %rhs_to_add : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%fa = %a : !fabric.bits<32>,
                %fb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
        %value = fabric.op [@arith.addi] (%fa, %fb)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %value : !fabric.bits<32>
      }
    }
    %tagged = fabric.boundary [s2t] %sum, %tag
        : (!fabric.bits<32>, !fabric.bits<4>) -> !fabric.bits_tag<32, 4>
    %remapped = fabric.boundary [t2t] %tagged
        {hw_params = [{lut_size = 4 : i32}],
         sw_configs = {lookup_table = [{src_tag = 0 : i4, dst_tag = 1 : i4}]}}
        : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 4>
    %untagged = fabric.boundary [t2s] %remapped
        : !fabric.bits_tag<32, 4> -> !fabric.bits<32>
    fabric.pe [spatial] (%a = %untagged : !fabric.bits<32>,
                         %b = %rhs_to_mul : !fabric.bits<32>,
                         %pc = %ctrl : !fabric.bits<0> to !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%fa = %a : !fabric.bits<32>,
                %fb = %b : !fabric.bits<32>,
                %token = %pc : !fabric.bits<32> to !fabric.bits<0>)
          -> !fabric.bits<32> {
        %value = fabric.op [@arith.muli] (%fa, %fb)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        %done, %published = fabric.op [@dataflow.sync] (%token, %value)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<32>)
              -> (!fabric.bits<0>, !fabric.bits<32>)
        fabric.yield %value : !fabric.bits<32>
      }
    }
    fabric.yield
  }

  fabric.module @boundary_t2s_tag_adg(%ctrl : !fabric.bits<0>,
                                      %lhs : !fabric.bits<32>,
                                      %rhs : !fabric.bits<32>,
                                      %tag : !fabric.bits<32>) {
    %rhs_to_add, %rhs_to_mul = fabric.switch [spatial] %rhs
        [{connectivity_table = ["1", "1"]}]
        : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
    %sum = fabric.pe [spatial] (%a = %lhs : !fabric.bits<32>,
                               %b = %rhs_to_add : !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%fa = %a : !fabric.bits<32>,
                %fb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
        %value = fabric.op [@arith.addi] (%fa, %fb)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %value : !fabric.bits<32>
      }
    }
    %tagged = fabric.boundary [s2t] %sum, %tag
        : (!fabric.bits<32>, !fabric.bits<32>)
       -> !fabric.bits_tag<32, 32>
    %discarded_data, %extracted_tag = fabric.boundary [t2s] %tagged
        : !fabric.bits_tag<32, 32>
       -> (!fabric.bits<32>, !fabric.bits<32>)
    fabric.pe [spatial] (%a = %extracted_tag : !fabric.bits<32>,
                         %b = %rhs_to_mul : !fabric.bits<32>,
                         %pc = %ctrl : !fabric.bits<0> to !fabric.bits<32>)
        -> !fabric.bits<32> {
      fabric.fu(%fa = %a : !fabric.bits<32>,
                %fb = %b : !fabric.bits<32>,
                %token = %pc : !fabric.bits<32> to !fabric.bits<0>)
          -> !fabric.bits<32> {
        %value = fabric.op [@arith.muli] (%fa, %fb)
            : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        %done, %published = fabric.op [@dataflow.sync] (%token, %value)
            {sw_configs = {bitmask = "11"}}
            : (!fabric.bits<0>, !fabric.bits<32>)
              -> (!fabric.bits<0>, !fabric.bits<32>)
        fabric.yield %value : !fabric.bits<32>
      }
    }
    fabric.yield
  }
}
