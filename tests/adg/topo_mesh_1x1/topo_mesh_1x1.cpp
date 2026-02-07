//===-- topo_mesh_1x1.cpp - ADG test: minimal 1x1 mesh ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("topo_mesh_1x1");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(5, 5)
      .setType(Type::i32());

  auto mesh = builder.buildMesh(1, 1, pe, sw, Topology::Mesh);

  // Single PE: module inputs -> PE[0][0] -> module output
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(a, mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(b, mesh.peGrid[0][0], 1);
  builder.connectToModuleOutput(mesh.peGrid[0][0], 0, out);

  builder.exportMLIR("Output/topo_mesh_1x1.fabric.mlir");
  return 0;
}
