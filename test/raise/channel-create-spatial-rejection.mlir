// RUN: not loom-raise-opt %s 2>&1 | FileCheck %s

// CHECK: 'dataflow.channel.create' op must not appear inside loom.spatial_region

module {
  dataflow.thread private @reject_spatial_create
      domain(#dataflow.thread_domain<dense>)() ctrl (%ctrl: none) {
    "loom.spatial_region"()
        <{operandSegmentSizes = array<i32: 0, 0, 0, 0>,
          resultSegmentSizes = array<i32: 0, 0>}> ({
      %channel = dataflow.channel.create : !dataflow.channel<i32>
      "loom.spatial_yield"()
          <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) {source_maps = []} : () -> ()
    dataflow.thread.yield
  }
}
