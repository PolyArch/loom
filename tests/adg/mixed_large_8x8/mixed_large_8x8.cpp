//===-- mixed_large_8x8.cpp - ADG test: 8x8 mesh stress test -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

#include <string>
#include <vector>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_large_8x8");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(5, 5)
      .setType(Type::i32());

  auto mesh = builder.buildMesh(8, 8, pe, sw, Topology::Mesh);

  // Create module inputs: two per PE (PE outputs already go to the co-located
  // switch via buildMesh, so no direct PE output chaining).
  std::vector<PortHandle> inputs;
  for (int i = 0; i < 128; ++i)
    inputs.push_back(
        builder.addModuleInput("in" + std::to_string(i), Type::i32()));

  // Connect each PE's two inputs from module inputs
  int idx = 0;
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      builder.connectToModuleInput(inputs[idx], mesh.peGrid[r][c], 0);
      builder.connectToModuleInput(inputs[idx + 1], mesh.peGrid[r][c], 1);
      idx += 2;
    }
  }

  builder.exportMLIR("Output/mixed_large_8x8.fabric.mlir");
  return 0;
}
