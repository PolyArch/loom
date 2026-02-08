//===-- switch_4x4_ring.cpp - ADG test: 4x4 ring connectivity ---*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("switch_4x4_ring");

  std::vector<std::vector<bool>> ring = {
      {true, true, false, false},
      {false, true, true, false},
      {false, false, true, true},
      {true, false, false, true}
  };

  auto sw = builder.newSwitch("ring_4x4")
      .setPortCount(4, 4)
      .setConnectivity(ring)
      .setType(Type::i32());

  auto inst = builder.clone(sw, "sw_0");

  auto in0 = builder.addModuleInput("a", Type::i32());
  auto in1 = builder.addModuleInput("b", Type::i32());
  auto in2 = builder.addModuleInput("c", Type::i32());
  auto in3 = builder.addModuleInput("d", Type::i32());
  auto out0 = builder.addModuleOutput("r0", Type::i32());
  auto out1 = builder.addModuleOutput("r1", Type::i32());
  auto out2 = builder.addModuleOutput("r2", Type::i32());
  auto out3 = builder.addModuleOutput("r3", Type::i32());

  for (int i = 0; i < 4; ++i)
    builder.connectToModuleInput(PortHandle{(unsigned)i}, inst, i);
  for (int i = 0; i < 4; ++i)
    builder.connectToModuleOutput(inst, i, PortHandle{(unsigned)(4 + i)});

  builder.exportMLIR("Output/switch_4x4_ring.fabric.mlir");
  return 0;
}
