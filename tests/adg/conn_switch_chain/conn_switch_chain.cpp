//===-- conn_switch_chain.cpp - ADG test: two switches in series -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_switch_chain");

  auto sw = builder.newSwitch("sw_2x2")
      .setPortCount(2, 2)
      .setType(Type::i32());

  auto sw0 = builder.clone(sw, "sw0");
  auto sw1 = builder.clone(sw, "sw1");

  auto x0 = builder.addModuleInput("x0", Type::i32());
  auto x1 = builder.addModuleInput("x1", Type::i32());
  auto out0 = builder.addModuleOutput("out0", Type::i32());
  auto out1 = builder.addModuleOutput("out1", Type::i32());

  builder.connectToModuleInput(x0, sw0, 0);
  builder.connectToModuleInput(x1, sw0, 1);
  builder.connectPorts(sw0, 0, sw1, 0);
  builder.connectPorts(sw0, 1, sw1, 1);
  builder.connectToModuleOutput(sw1, 0, out0);
  builder.connectToModuleOutput(sw1, 1, out1);

  builder.exportMLIR("Output/conn_switch_chain.fabric.mlir");
  return 0;
}
