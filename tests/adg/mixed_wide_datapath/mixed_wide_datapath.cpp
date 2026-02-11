//===-- mixed_wide_datapath.cpp - ADG test: 8-port switch + 4 PEs -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_wide_datapath");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  // First switch: distributes module inputs to first-stage PEs
  auto sw1 = builder.newSwitch("dist_xbar")
      .setPortCount(4, 4)
      .setType(Type::i32());

  // Second switch: distributes first-stage PE outputs to second-stage PEs
  auto sw2 = builder.newSwitch("gather_xbar")
      .setPortCount(4, 4)
      .setType(Type::i32());

  auto sw1_0 = builder.clone(sw1, "sw1_0");
  auto pe0 = builder.clone(pe, "pe0");
  auto pe1 = builder.clone(pe, "pe1");
  auto sw2_0 = builder.clone(sw2, "sw2_0");
  auto pe2 = builder.clone(pe, "pe2");
  auto pe3 = builder.clone(pe, "pe3");

  // Module I/O
  auto in0 = builder.addModuleInput("in0", Type::i32());
  auto in1 = builder.addModuleInput("in1", Type::i32());
  auto in2 = builder.addModuleInput("in2", Type::i32());
  auto in3 = builder.addModuleInput("in3", Type::i32());
  auto in4 = builder.addModuleInput("in4", Type::i32());
  auto in5 = builder.addModuleInput("in5", Type::i32());
  auto out0 = builder.addModuleOutput("out0", Type::i32());
  auto out1 = builder.addModuleOutput("out1", Type::i32());
  auto out2 = builder.addModuleOutput("out2", Type::i32());
  auto out3 = builder.addModuleOutput("out3", Type::i32());

  // First switch: all 4 inputs from module
  builder.connectToModuleInput(in0, sw1_0, 0);
  builder.connectToModuleInput(in1, sw1_0, 1);
  builder.connectToModuleInput(in2, sw1_0, 2);
  builder.connectToModuleInput(in3, sw1_0, 3);

  // First switch outputs -> first-stage PEs
  builder.connectPorts(sw1_0, 0, pe0, 0);
  builder.connectPorts(sw1_0, 1, pe0, 1);
  builder.connectPorts(sw1_0, 2, pe1, 0);
  builder.connectPorts(sw1_0, 3, pe1, 1);

  // Second switch: first-stage PE outputs + additional module inputs
  auto bcast_0_sw_def = builder.newSwitch("bcast_0_sw")
      .setPortCount(1, 2)
      .setType(Type::i32());
  auto bcast_0 = builder.clone(bcast_0_sw_def, "bcast_0");
  builder.connectPorts(pe0, 0, bcast_0, 0);
  builder.connectPorts(bcast_0, 0, sw2_0, 0);
  builder.connectToModuleOutput(bcast_0, 1, out0);
  auto bcast_1_sw_def = builder.newSwitch("bcast_1_sw")
      .setPortCount(1, 2)
      .setType(Type::i32());
  auto bcast_1 = builder.clone(bcast_1_sw_def, "bcast_1");
  builder.connectPorts(pe1, 0, bcast_1, 0);
  builder.connectPorts(bcast_1, 0, sw2_0, 1);
  builder.connectToModuleOutput(bcast_1, 1, out1);
  builder.connectToModuleInput(in4, sw2_0, 2);
  builder.connectToModuleInput(in5, sw2_0, 3);

  // Second switch outputs -> second-stage PEs
  builder.connectPorts(sw2_0, 0, pe2, 0);
  builder.connectPorts(sw2_0, 1, pe2, 1);
  builder.connectPorts(sw2_0, 2, pe3, 0);
  builder.connectPorts(sw2_0, 3, pe3, 1);

  // PE outputs -> module outputs
  // Also need module inputs for unused switch outputs or direct PE connections
  builder.connectToModuleOutput(pe2, 0, out2);
  builder.connectToModuleOutput(pe3, 0, out3);

  builder.exportMLIR("Output/mixed_wide_datapath.fabric.mlir");
  return 0;
}
