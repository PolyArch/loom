//===-- temporal_sw_4x4.cpp - ADG test: 4x4 temporal switch ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("temporal_sw_4x4");

  auto taggedType = Type::tagged(Type::i32(), Type::iN(4));

  auto tsw = builder.newTemporalSwitch("tsw_4x4")
      .setNumRouteTable(4)
      .setPortCount(4, 4)
      .setInterface(taggedType);

  auto inst = builder.clone(tsw, "tsw0");

  auto in0 = builder.addModuleInput("a", taggedType);
  auto in1 = builder.addModuleInput("b", taggedType);
  auto in2 = builder.addModuleInput("c", taggedType);
  auto in3 = builder.addModuleInput("d", taggedType);
  auto out0 = builder.addModuleOutput("r0", taggedType);
  auto out1 = builder.addModuleOutput("r1", taggedType);
  auto out2 = builder.addModuleOutput("r2", taggedType);
  auto out3 = builder.addModuleOutput("r3", taggedType);

  for (int i = 0; i < 4; ++i)
    builder.connectToModuleInput(PortHandle{(unsigned)i}, inst, i);
  for (int i = 0; i < 4; ++i)
    builder.connectToModuleOutput(inst, i, PortHandle{(unsigned)(4 + i)});

  builder.exportMLIR("Output/temporal_sw_4x4.fabric.mlir");
  return 0;
}
