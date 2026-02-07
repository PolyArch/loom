//===-- topo_mesh_tagged.cpp - ADG test: mesh with tagged types -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("topo_mesh_tagged");

  auto taggedType = Type::tagged(Type::i32(), Type::iN(4));

  auto pe = builder.newPE("tagged_alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInterfaceCategory(InterfaceCategory::Tagged)
      .setInputPorts({taggedType, taggedType})
      .setOutputPorts({taggedType})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("tagged_xbar")
      .setPortCount(5, 5)
      .setType(taggedType);

  auto mesh = builder.buildMesh(2, 2, pe, sw, Topology::Mesh);

  // Chain PEs in row-major order: [0][0] -> [0][1] -> [1][0] -> [1][1]
  auto a = builder.addModuleInput("a", taggedType);
  auto b = builder.addModuleInput("b", taggedType);
  auto c = builder.addModuleInput("c", taggedType);
  auto d = builder.addModuleInput("d", taggedType);
  auto e = builder.addModuleInput("e", taggedType);
  auto out = builder.addModuleOutput("result", taggedType);

  builder.connectToModuleInput(a, mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(b, mesh.peGrid[0][0], 1);
  builder.connectPorts(mesh.peGrid[0][0], 0, mesh.peGrid[0][1], 0);
  builder.connectToModuleInput(c, mesh.peGrid[0][1], 1);
  builder.connectPorts(mesh.peGrid[0][1], 0, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(d, mesh.peGrid[1][0], 1);
  builder.connectPorts(mesh.peGrid[1][0], 0, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(e, mesh.peGrid[1][1], 1);
  builder.connectToModuleOutput(mesh.peGrid[1][1], 0, out);

  builder.exportMLIR("Output/topo_mesh_tagged.fabric.mlir");
  return 0;
}
