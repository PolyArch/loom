//===-- conn_switch_fanout.cpp - ADG test: switch 1-to-4 fanout -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_switch_fanout");

  auto sw = builder.newSwitch("sw_1x4")
      .setPortCount(1, 4)
      .setType(Type::i32());

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw0 = builder.clone(sw, "sw0");
  auto a0 = builder.clone(adder, "a0");
  auto a1 = builder.clone(adder, "a1");

  auto x = builder.addModuleInput("x", Type::i32());
  auto y = builder.addModuleInput("y", Type::i32());
  auto out0 = builder.addModuleOutput("out0", Type::i32());
  auto out1 = builder.addModuleOutput("out1", Type::i32());

  builder.connectToModuleInput(x, sw0, 0);
  builder.connectPorts(sw0, 0, a0, 0);
  builder.connectPorts(sw0, 1, a0, 1);
  builder.connectPorts(sw0, 2, a1, 0);
  builder.connectPorts(sw0, 3, a1, 1);
  builder.connectToModuleInput(y, a1, 1); // override sw0 port 3
  builder.connectToModuleOutput(a0, 0, out0);
  builder.connectToModuleOutput(a1, 0, out1);

  builder.exportMLIR("Output/conn_switch_fanout.fabric.mlir");
  return 0;
}
