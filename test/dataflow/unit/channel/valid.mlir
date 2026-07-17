// RUN: loom %s | loom | FileCheck %s
// RUN: loom %s --canonicalize | FileCheck %s --check-prefix=EFFECT

// CHECK-LABEL: dataflow.thread private @relay_channel
dataflow.thread private @relay_channel(
    %channel: !dataflow.channel<i32>, %enabled: i1) ctrl (%ctrl: none) {
  scf.if %enabled {
    // CHECK: %{{.*}} = dataflow.channel.receive %{{.*}} : !dataflow.channel<i32>
    %message = dataflow.channel.receive %channel : !dataflow.channel<i32>
    // CHECK: dataflow.channel.send %{{.*}}, %{{.*}} : !dataflow.channel<i32>
    dataflow.channel.send %channel, %message : !dataflow.channel<i32>
  }
  dataflow.thread.yield
}

// EFFECT-LABEL: dataflow.thread private @relay_channel
// EFFECT: dataflow.channel.receive
// EFFECT: dataflow.channel.send

dataflow.thread private @receive_unused(
    %channel: !dataflow.channel<i32>) ctrl (%ctrl: none) {
  %message = dataflow.channel.receive %channel : !dataflow.channel<i32>
  dataflow.thread.yield
}

// EFFECT-LABEL: dataflow.thread private @receive_unused
// EFFECT: %{{.*}} = dataflow.channel.receive %{{.*}} : !dataflow.channel<i32>

func.func @launch_relay(%channel: !dataflow.channel<i32>, %enabled: i1) {
  // CHECK: %{{.*}} = dataflow.thread.launch @relay_channel(%{{.*}}, %{{.*}}) : (!dataflow.channel<i32>, i1) -> !dataflow.thread_token
  %completion = dataflow.thread.launch @relay_channel(%channel, %enabled)
      : (!dataflow.channel<i32>, i1) -> !dataflow.thread_token
  return
}
