//===-- mixed_large_8x8.cpp - ADG test: 8x8 mesh stress test -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

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

  // Create module inputs: in0 (first input to PE[0][0]), and one per PE for
  // the second input port (in1 through in64)
  std::vector<PortHandle> inputs;
  for (int i = 0; i <= 64; ++i)
    inputs.push_back(
        builder.addModuleInput("in" + std::to_string(i), Type::i32()));
  auto result = builder.addModuleOutput("result", Type::i32());

  // First PE: both inputs from module
  builder.connectToModuleInput(inputs[0], mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(inputs[1], mesh.peGrid[0][0], 1);

  // Chain remaining PEs in row-major order
  int idx = 2;
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      if (r == 0 && c == 0)
        continue; // already connected
      // Determine previous PE in row-major order
      int prevR = (c == 0) ? r - 1 : r;
      int prevC = (c == 0) ? 7 : c - 1;
      builder.connectPorts(mesh.peGrid[prevR][prevC], 0,
                           mesh.peGrid[r][c], 0);
      builder.connectToModuleInput(inputs[idx], mesh.peGrid[r][c], 1);
      ++idx;
    }
  }

  builder.connectToModuleOutput(mesh.peGrid[7][7], 0, result);

  builder.exportMLIR("Output/mixed_large_8x8.fabric.mlir");
  return 0;
}
