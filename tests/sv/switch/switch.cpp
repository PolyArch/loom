//===-- switch.cpp - SV test: single switch module ------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("switch");

  auto sw = builder.newSwitch("mux")
      .setPortCount(2, 2)
      .setType(Type::i32());

  auto sw0 = builder.clone(sw, "sw0");

  auto in0 = builder.addModuleInput("in0", Type::i32());
  auto in1 = builder.addModuleInput("in1", Type::i32());
  auto out0 = builder.addModuleOutput("out0", Type::i32());
  auto out1 = builder.addModuleOutput("out1", Type::i32());

  builder.connectToModuleInput(in0, sw0, 0);
  builder.connectToModuleInput(in1, sw0, 1);
  builder.connectToModuleOutput(sw0, 0, out0);
  builder.connectToModuleOutput(sw0, 1, out1);

  builder.exportMLIR("Output/switch.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
