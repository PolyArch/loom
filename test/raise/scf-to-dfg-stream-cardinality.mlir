// RUN: loom-raise-opt --loom-lower-for-to-graph %s -o %t.lowered.mlir
// RUN: FileCheck %s < %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph wide_stream_schedule_graph --arg 0=2 --arg 1=11 --arg 2=13 --arg 3=17 --output %t.result.json
// RUN: FileCheck %s --check-prefix=SIM < %t.result.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph wide_integer_stream_schedule_graph --arg 0=2 --arg 1=19 --output %t.integer-result.json
// RUN: FileCheck %s --check-prefix=INTEGER-SIM < %t.integer-result.json

// CHECK-LABEL: dataflow.graph private @wide_stream_schedule_graph(
// CHECK: arith.index_cast {{.*}} : index to i6
// CHECK: dataflow.stream
// CHECK-SAME: step add while ult : i6
// CHECK-NOT: dataflow.channel
// CHECK: dataflow.graph.return values() streams(%{{.*}} : i32)

// CHECK-LABEL: dataflow.graph private @wide_integer_stream_schedule_graph(
// CHECK: arith.extsi {{.*}} : i64 to i65
// CHECK: dataflow.stream
// CHECK-SAME: step add while ult : i65
// CHECK-NOT: dataflow.channel
// CHECK: dataflow.graph.return values() streams(%{{.*}} : i32)

// SIM: "final_stream_outputs": [
// SIM-NEXT: [
// SIM-NEXT: "i32:11",
// SIM-NEXT: "i32:13",
// SIM-NEXT: "i32:13",
// SIM-NEXT: "i32:13",
// SIM-NEXT: "i32:13",
// SIM-NEXT: "i32:17"
// SIM: "status": "pass"

// INTEGER-SIM: "final_stream_outputs": [
// INTEGER-SIM-NEXT: [
// INTEGER-SIM-NEXT: "i32:19",
// INTEGER-SIM-NEXT: "i32:19",
// INTEGER-SIM-NEXT: "i32:19",
// INTEGER-SIM-NEXT: "i32:19",
// INTEGER-SIM-NEXT: "i32:19"
// INTEGER-SIM: "status": "pass"

module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 4>>
} {
  dataflow.thread private @wide_stream_schedule domain(#dataflow.thread_domain<dense>)(
      %output: !dataflow.channel<i32>, %count: index, %prefix: i32,
      %body: i32, %suffix: i32) ctrl (%ctrl: none) {
    "loom.spatial_region"(%count, %prefix, %body, %suffix, %output)
        <{operandSegmentSizes = array<i32: 4, 0, 0, 1>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%limit: index, %before: i32, %message: i32, %after: i32,
           %channel: !dataflow.channel<i32>):
        dataflow.channel.send %channel, %before : !dataflow.channel<i32>
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        scf.for %iv = %zero to %limit step %one {
          dataflow.channel.send %channel, %message : !dataflow.channel<i32>
          dataflow.channel.send %channel, %message : !dataflow.channel<i32>
        }
        dataflow.channel.send %channel, %after : !dataflow.channel<i32>
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "wide_stream_schedule_graph", source_maps = []} :
        (index, i32, i32, i32, !dataflow.channel<i32>) -> ()
    dataflow.thread.yield
  }

  dataflow.thread private @wide_integer_stream_schedule domain(#dataflow.thread_domain<dense>)(
      %output: !dataflow.channel<i32>, %count: i64, %body: i32)
      ctrl (%ctrl: none) {
    "loom.spatial_region"(%count, %body, %output)
        <{operandSegmentSizes = array<i32: 2, 0, 0, 1>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      ^bb0(%limit: i64, %message: i32,
           %channel: !dataflow.channel<i32>):
        dataflow.channel.send %channel, %message : !dataflow.channel<i32>
        %zero = arith.constant 0 : i64
        %one = arith.constant 1 : i64
        scf.for %iv = %zero to %limit step %one : i64 {
          dataflow.channel.send %channel, %message : !dataflow.channel<i32>
          dataflow.channel.send %channel, %message : !dataflow.channel<i32>
        }
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {graph_name = "wide_integer_stream_schedule_graph", source_maps = []} :
        (i64, i32, !dataflow.channel<i32>) -> ()
    dataflow.thread.yield
  }
}
