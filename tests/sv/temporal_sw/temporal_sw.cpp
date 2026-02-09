//===-- temporal_sw.cpp - SV test: temporal switch module ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("temporal_sw");

  auto tsw = builder.newTemporalSwitch("router")
      .setPortCount(2, 2)
      .setNumRouteTable(4)
      .setInterface(Type::tagged(Type::i32(), Type::iN(4)));

  auto t0 = builder.clone(tsw, "t0");

  auto in0 = builder.addModuleInput("in0", Type::tagged(Type::i32(), Type::iN(4)));
  auto in1 = builder.addModuleInput("in1", Type::tagged(Type::i32(), Type::iN(4)));
  auto out0 = builder.addModuleOutput("out0", Type::tagged(Type::i32(), Type::iN(4)));
  auto out1 = builder.addModuleOutput("out1", Type::tagged(Type::i32(), Type::iN(4)));

  builder.connectToModuleInput(in0, t0, 0);
  builder.connectToModuleInput(in1, t0, 1);
  builder.connectToModuleOutput(t0, 0, out0);
  builder.connectToModuleOutput(t0, 1, out1);

  builder.exportMLIR("Output/temporal_sw.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
