// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-graph-memory %t.dir/valid.mlir | FileCheck %s --check-prefixes=ONE,MULTI --implicit-check-not='scf.'
// RUN: not loom-raise-opt --mlir-disable-threading --loom-lower-graph-memory --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/zero-case.mlir 2>&1 | FileCheck %s --check-prefix=ZERO --implicit-check-not='arith.cmpi' --implicit-check-not='arith.select' --implicit-check-not='dataflow.constant' --implicit-check-not='dataflow.mux' --implicit-check-not='dataflow.demux' --implicit-check-not='dataflow.load' --implicit-check-not='dataflow.store'

//--- valid.mlir
module {
  // ONE-LABEL: dataflow.graph private @index_switch_one_case
  // ONE: %[[CASE:.*]] = dataflow.constant %arg0 {const_value = 7 : index} : index
  // ONE: %[[SELECTOR:.*]] = arith.cmpi eq, %arg1, %[[CASE]] : index
  // ONE: %[[EXECUTION:.*]]:2 = dataflow.demux %[[SELECTOR]], %arg0 : (i1, none) -> (none, none)
  // ONE-DAG: %[[DEFAULT_VALUE:.*]]:2 = dataflow.demux %[[SELECTOR]], %arg2 : (i1, i32) -> (i32, i32)
  // ONE-DAG: %[[CASE_VALUE:.*]]:2 = dataflow.demux %[[SELECTOR]], %arg3 : (i1, i32) -> (i32, i32)
  // ONE: %[[RESULT:.*]] = dataflow.mux %[[SELECTOR]], %[[DEFAULT_VALUE]]#0, %[[CASE_VALUE]]#1 : (i1, i32, i32) -> i32
  // ONE: %[[COMPLETE:.*]] = dataflow.mux %[[SELECTOR]], %[[EXECUTION]]#0, %[[EXECUTION]]#1 : (i1, none, none) -> none
  // ONE: %[[PUBLISHED:.*]]:2 = dataflow.sync %[[COMPLETE]], %[[RESULT]] : (none, i32) -> (none, i32)
  // ONE: dataflow.graph.return %[[PUBLISHED]]#0, %[[PUBLISHED]]#1 : none, i32
  dataflow.graph private @index_switch_one_case(
      %start: none, %selector: index, %default_value: i32,
      %case_value: i32) -> (i32) {
    %selected = scf.index_switch %selector -> i32
    case 7 {
      scf.yield %case_value : i32
    }
    default {
      scf.yield %default_value : i32
    }
    dataflow.graph.return %start, %selected : none, i32
  }

  // MULTI-LABEL: dataflow.graph private @index_switch_multiple_cases
  // MULTI: %[[LANE_ZERO:.*]] = dataflow.constant %arg0 {const_value = 0 : index} : index
  // MULTI: %[[CASE_TWO:.*]] = dataflow.constant %arg0 {const_value = 2 : index} : index
  // MULTI: %[[MATCH_TWO:.*]] = arith.cmpi eq, %arg1, %[[CASE_TWO]] : index
  // MULTI: %[[LANE_ONE:.*]] = dataflow.constant %arg0 {const_value = 1 : index} : index
  // MULTI: %[[AFTER_TWO:.*]] = arith.select %[[MATCH_TWO]], %[[LANE_ONE]], %[[LANE_ZERO]] : index
  // MULTI: %[[CASE_FIVE:.*]] = dataflow.constant %arg0 {const_value = 5 : index} : index
  // MULTI: %[[MATCH_FIVE:.*]] = arith.cmpi eq, %arg1, %[[CASE_FIVE]] : index
  // MULTI: %[[LANE_TWO:.*]] = dataflow.constant %arg0 {const_value = 2 : index} : index
  // MULTI: %[[LANE:.*]] = arith.select %[[MATCH_FIVE]], %[[LANE_TWO]], %[[AFTER_TWO]] : index
  // MULTI: %[[EXECUTION:.*]]:3 = dataflow.demux %[[LANE]], %arg0 : (index, none) -> (none, none, none)
  // MULTI: %[[WRITE:.*]]:3 = dataflow.demux %[[LANE]], %arg0 : (index, none) -> (none, none, none)
  // MULTI: %[[READ:.*]]:3 = dataflow.demux %[[LANE]], %arg0 : (index, none) -> (none, none, none)
  // MULTI-DAG: %[[INDEX:.*]]:3 = dataflow.demux %[[LANE]], %arg2 : (index, index) -> (index, index, index)
  // MULTI-DAG: %[[DEFAULT_FIRST:.*]]:3 = dataflow.demux %[[LANE]], %arg3 : (index, i32) -> (i32, i32, i32)
  // MULTI-DAG: %[[DEFAULT_SECOND:.*]]:3 = dataflow.demux %[[LANE]], %arg4 : (index, i32) -> (i32, i32, i32)
  // MULTI-DAG: %[[CASE_TWO_VALUE:.*]]:3 = dataflow.demux %[[LANE]], %arg5 : (index, i32) -> (i32, i32, i32)
  // MULTI-DAG: %[[CASE_FIVE_VALUE:.*]]:3 = dataflow.demux %[[LANE]], %arg6 : (index, i32) -> (i32, i32, i32)
  // MULTI: %[[CASE_TWO_CTRL:.*]]:2 = dataflow.sync %[[EXECUTION]]#1, %[[READ]]#1 : (none, none) -> (none, none)
  // MULTI: %[[CASE_TWO_DONE:.*]] = dataflow.store %arg7[%[[INDEX]]#1] %[[CASE_TWO_VALUE]]#1 %[[CASE_TWO_CTRL]]#0 : memref<?xi32>
  // MULTI: %[[CASE_FIVE_CTRL:.*]]:2 = dataflow.sync %[[EXECUTION]]#2, %[[WRITE]]#2 : (none, none) -> (none, none)
  // MULTI: %[[CASE_FIVE_DATA:.*]], %[[CASE_FIVE_DONE:.*]] = dataflow.load %arg7[%[[INDEX]]#2] %[[CASE_FIVE_CTRL]]#0 : memref<?xi32>
  // MULTI: %[[CASE_FIVE_READ:.*]]:2 = dataflow.sync %[[READ]]#2, %[[CASE_FIVE_DONE]] : (none, none) -> (none, none)
  // MULTI: %[[FIRST:.*]] = dataflow.mux %[[LANE]], %[[DEFAULT_FIRST]]#0, %[[CASE_TWO_VALUE]]#1, %[[CASE_FIVE_DATA]] : (index, i32, i32, i32) -> i32
  // MULTI: %[[SECOND:.*]] = dataflow.mux %[[LANE]], %[[DEFAULT_SECOND]]#0, %[[DEFAULT_SECOND]]#1, %[[CASE_FIVE_VALUE]]#2 : (index, i32, i32, i32) -> i32
  // MULTI: %[[WRITE_OUT:.*]] = dataflow.mux %[[LANE]], %[[WRITE]]#0, %[[CASE_TWO_DONE]], %[[WRITE]]#2 : (index, none, none, none) -> none
  // MULTI: %[[READ_OUT:.*]] = dataflow.mux %[[LANE]], %[[READ]]#0, %[[CASE_TWO_DONE]], %[[CASE_FIVE_READ]]#0 : (index, none, none, none) -> none
  // MULTI: %[[EXECUTION_OUT:.*]] = dataflow.mux %[[LANE]], %[[EXECUTION]]#0, %[[EXECUTION]]#1, %[[EXECUTION]]#2 : (index, none, none, none) -> none
  // MULTI: %[[AFTER_CTRL:.*]]:2 = dataflow.sync %[[EXECUTION_OUT]], %[[READ_OUT]] : (none, none) -> (none, none)
  // MULTI: %[[AFTER_DONE:.*]] = dataflow.store %arg7[%arg2] %[[FIRST]] %[[AFTER_CTRL]]#0 : memref<?xi32>
  // MULTI: dataflow.sync %[[AFTER_DONE]], %[[FIRST]] : (none, i32) -> (none, i32)
  // MULTI: dataflow.sync %[[AFTER_DONE]], %[[SECOND]] : (none, i32) -> (none, i32)
  dataflow.graph private @index_switch_multiple_cases(
      %start: none, %selector: index, %index: index,
      %default_first: i32, %default_second: i32,
      %case_two_value: i32, %case_five_value: i32,
      %memory: memref<?xi32>) -> (i32, i32)
      attributes {input_segments = array<i32: 6, 0, 1>,
                  result_segments = array<i32: 2, 0, 0>} {
    %first, %second = scf.index_switch %selector -> i32, i32
    case 2 {
      memref.store %case_two_value, %memory[%index] : memref<?xi32>
      scf.yield %case_two_value, %default_second : i32, i32
    }
    case 5 {
      %loaded = memref.load %memory[%index] : memref<?xi32>
      scf.yield %loaded, %case_five_value : i32, i32
    }
    default {
      scf.yield %default_first, %default_second : i32, i32
    }
    memref.store %first, %memory[%index] : memref<?xi32>
    dataflow.graph.return %start, %first, %second : none, i32, i32
  }
}

//--- zero-case.mlir
module {
  // ZERO: error: loom-lower-graph-memory: zero-case scf.index_switch requires upstream normalization before graph-region lowering
  // ZERO-LABEL: // -----// IR Dump After
  // ZERO: dataflow.graph private @index_switch_zero_case
  // ZERO: %[[SELECTED:.*]] = scf.index_switch %arg1 -> i32
  // ZERO: default {
  // ZERO: %[[LOADED:.*]] = memref.load %arg3[%arg2] : memref<?xi32>
  // ZERO: scf.yield %[[LOADED]] : i32
  // ZERO: dataflow.graph.return %arg0, %[[SELECTED]] : none, i32
  dataflow.graph private @index_switch_zero_case(
      %start: none, %selector: index, %index: index,
      %memory: memref<?xi32>) -> (i32)
      attributes {input_segments = array<i32: 2, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %selected = scf.index_switch %selector -> i32
    default {
      %loaded = memref.load %memory[%index] : memref<?xi32>
      scf.yield %loaded : i32
    }
    dataflow.graph.return %start, %selected : none, i32
  }
}
