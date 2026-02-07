//===-- conn_connect_simple.cpp - ADG test: connect() shorthand -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_connect_simple");

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
  builder.connect(sw0, sw1); // port 0 -> port 0 shorthand
  builder.connectPorts(sw0, 1, sw1, 1);
  builder.connectToModuleOutput(sw1, 0, out0);
  builder.connectToModuleOutput(sw1, 1, out1);

  builder.exportMLIR("Output/conn_connect_simple.fabric.mlir");
  return 0;
}
