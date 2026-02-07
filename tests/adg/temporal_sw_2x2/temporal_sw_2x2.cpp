//===-- temporal_sw_2x2.cpp - ADG test: 2x2 temporal switch ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("temporal_sw_2x2");

  auto taggedType = Type::tagged(Type::i32(), Type::iN(4));

  auto tsw = builder.newTemporalSwitch("tsw_2x2")
      .setNumRouteTable(2)
      .setPortCount(2, 2)
      .setInterface(taggedType);

  auto inst = builder.clone(tsw, "tsw0");

  auto in0 = builder.addModuleInput("a", taggedType);
  auto in1 = builder.addModuleInput("b", taggedType);
  auto out0 = builder.addModuleOutput("r0", taggedType);
  auto out1 = builder.addModuleOutput("r1", taggedType);

  builder.connectToModuleInput(in0, inst, 0);
  builder.connectToModuleInput(in1, inst, 1);
  builder.connectToModuleOutput(inst, 0, out0);
  builder.connectToModuleOutput(inst, 1, out1);

  builder.exportMLIR("Output/temporal_sw_2x2.fabric.mlir");
  return 0;
}
