//===-- switch_tagged.cpp - ADG test: switch with tagged types ---*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("switch_tagged");

  auto taggedType = Type::tagged(Type::i32(), Type::iN(4));

  auto sw = builder.newSwitch("tagged_sw_2x2")
      .setPortCount(2, 2)
      .setType(taggedType);

  auto inst = builder.clone(sw, "sw_0");

  auto in0 = builder.addModuleInput("a", taggedType);
  auto in1 = builder.addModuleInput("b", taggedType);
  auto out0 = builder.addModuleOutput("r0", taggedType);
  auto out1 = builder.addModuleOutput("r1", taggedType);

  builder.connectToModuleInput(in0, inst, 0);
  builder.connectToModuleInput(in1, inst, 1);
  builder.connectToModuleOutput(inst, 0, out0);
  builder.connectToModuleOutput(inst, 1, out1);

  builder.exportMLIR("Output/switch_tagged.fabric.mlir");
  return 0;
}
