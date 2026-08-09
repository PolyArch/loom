// RUN: loom %s | loom | FileCheck %s
// RUN: loom %s --canonicalize | FileCheck %s --check-prefix=EFFECT

// CHECK-LABEL: dataflow.thread private @relay_channel domain(#dataflow.thread_domain<dense>)
dataflow.thread private @relay_channel domain(#dataflow.thread_domain<dense>)(
    %channel: !dataflow.channel<i32>, %enabled: i1) ctrl (%ctrl: none) {
  scf.if %enabled {
    // CHECK: %{{.*}} = dataflow.channel.receive %{{.*}} : !dataflow.channel<i32>
    %message = dataflow.channel.receive %channel : !dataflow.channel<i32>
    // CHECK: dataflow.channel.send %{{.*}}, %{{.*}} : !dataflow.channel<i32>
    dataflow.channel.send %channel, %message : !dataflow.channel<i32>
  }
  dataflow.thread.yield
}

// EFFECT-LABEL: dataflow.thread private @relay_channel domain(#dataflow.thread_domain<dense>)
// EFFECT: dataflow.channel.receive
// EFFECT: dataflow.channel.send

dataflow.thread private @receive_unused domain(#dataflow.thread_domain<dense>)(
    %channel: !dataflow.channel<i32>) ctrl (%ctrl: none) {
  %message = dataflow.channel.receive %channel : !dataflow.channel<i32>
  dataflow.thread.yield
}

// EFFECT-LABEL: dataflow.thread private @receive_unused domain(#dataflow.thread_domain<dense>)
// EFFECT: %{{.*}} = dataflow.channel.receive %{{.*}} : !dataflow.channel<i32>

dataflow.thread private @created_channel_producer
    domain(#dataflow.thread_domain<dense>)(
        %channel: !dataflow.channel<i32>) ctrl (%ctrl: none) {
  %message = arith.constant 7 : i32
  dataflow.channel.send %channel, %message : !dataflow.channel<i32>
  dataflow.thread.yield
}

dataflow.thread private @created_channel_consumer
    domain(#dataflow.thread_domain<dense>)(
        %channel: !dataflow.channel<i32>) ctrl (%ctrl: none) {
  %message = dataflow.channel.receive %channel : !dataflow.channel<i32>
  dataflow.thread.yield
}

func.func @launch_created_channel() {
  // CHECK: %[[CHANNEL:.*]] = dataflow.channel.create : !dataflow.channel<i32>
  %channel = dataflow.channel.create : !dataflow.channel<i32>
  // CHECK: dataflow.thread.launch @created_channel_producer(%[[CHANNEL]])
  %producer = dataflow.thread.launch @created_channel_producer(%channel)
      : (!dataflow.channel<i32>) -> !dataflow.thread_token
  // CHECK: dataflow.thread.launch @created_channel_consumer(%[[CHANNEL]])
  %consumer = dataflow.thread.launch @created_channel_consumer(%channel)
      : (!dataflow.channel<i32>) -> !dataflow.thread_token
  return
}

func.func @launch_relay(%channel: !dataflow.channel<i32>, %enabled: i1) {
  // CHECK: %{{.*}} = dataflow.thread.launch @relay_channel(%{{.*}}, %{{.*}}) : (!dataflow.channel<i32>, i1) -> !dataflow.thread_token
  %completion = dataflow.thread.launch @relay_channel(%channel, %enabled)
      : (!dataflow.channel<i32>, i1) -> !dataflow.thread_token
  return
}

// A pointer is an ordinary stream payload, not a memory capability.
// CHECK-LABEL: dataflow.thread private @relay_pointer
dataflow.thread private @relay_pointer domain(#dataflow.thread_domain<dense>)(
    %channel: !dataflow.channel<!llvm.ptr>) ctrl (%ctrl: none) {
  %message = dataflow.channel.receive %channel : !dataflow.channel<!llvm.ptr>
  dataflow.channel.send %channel, %message : !dataflow.channel<!llvm.ptr>
  dataflow.thread.yield
}
