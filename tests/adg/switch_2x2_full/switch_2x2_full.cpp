//===-- switch_2x2_full.cpp - ADG test: 2x2 full crossbar switch -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("switch_2x2_full");

  auto sw = builder.newSwitch("xbar_2x2")
      .setPortCount(2, 2)
      .setType(Type::i32());

  auto inst = builder.clone(sw, "sw_0");

  auto in0 = builder.addModuleInput("a", Type::i32());
  auto in1 = builder.addModuleInput("b", Type::i32());
  auto out0 = builder.addModuleOutput("r0", Type::i32());
  auto out1 = builder.addModuleOutput("r1", Type::i32());

  builder.connectToModuleInput(in0, inst, 0);
  builder.connectToModuleInput(in1, inst, 1);
  builder.connectToModuleOutput(inst, 0, out0);
  builder.connectToModuleOutput(inst, 1, out1);

  builder.exportMLIR("Output/switch_2x2_full.fabric.mlir");
  return 0;
}
