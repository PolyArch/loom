//===-- temporal_sw_custom.cpp - ADG test: custom temporal sw --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("temporal_sw_custom");

  auto taggedType = Type::tagged(Type::i32(), Type::iN(4));

  std::vector<std::vector<bool>> conn = {
      {true, true, false},
      {false, true, true},
      {true, false, true}
  };

  auto tsw = builder.newTemporalSwitch("tsw_custom")
      .setNumRouteTable(4)
      .setPortCount(3, 3)
      .setConnectivity(conn)
      .setInterface(taggedType);

  auto inst = builder.clone(tsw, "tsw0");

  auto in0 = builder.addModuleInput("a", taggedType);
  auto in1 = builder.addModuleInput("b", taggedType);
  auto in2 = builder.addModuleInput("c", taggedType);
  auto out0 = builder.addModuleOutput("r0", taggedType);
  auto out1 = builder.addModuleOutput("r1", taggedType);
  auto out2 = builder.addModuleOutput("r2", taggedType);

  for (int i = 0; i < 3; ++i)
    builder.connectToModuleInput(PortHandle{(unsigned)i}, inst, i);
  for (int i = 0; i < 3; ++i)
    builder.connectToModuleOutput(inst, i, PortHandle{(unsigned)(3 + i)});

  builder.exportMLIR("Output/temporal_sw_custom.fabric.mlir");
  return 0;
}
