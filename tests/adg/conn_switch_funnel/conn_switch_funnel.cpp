//===-- conn_switch_funnel.cpp - ADG test: switch 4-to-1 funnel -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_switch_funnel");

  auto sw = builder.newSwitch("sw_4x1")
      .setPortCount(4, 1)
      .setType(Type::i32());

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto a0 = builder.clone(adder, "a0");
  auto a1 = builder.clone(adder, "a1");
  auto sw0 = builder.clone(sw, "sw0");

  auto x0 = builder.addModuleInput("x0", Type::i32());
  auto x1 = builder.addModuleInput("x1", Type::i32());
  auto x2 = builder.addModuleInput("x2", Type::i32());
  auto x3 = builder.addModuleInput("x3", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(x0, a0, 0);
  builder.connectToModuleInput(x1, a0, 1);
  builder.connectToModuleInput(x2, a1, 0);
  builder.connectToModuleInput(x3, a1, 1);
  builder.connectPorts(a0, 0, sw0, 0);
  builder.connectPorts(a1, 0, sw0, 1);
  builder.connectToModuleInput(x0, sw0, 2);
  builder.connectToModuleInput(x1, sw0, 3);
  builder.connectToModuleOutput(sw0, 0, out);

  builder.exportMLIR("Output/conn_switch_funnel.fabric.mlir");
  return 0;
}
