//===-- topo_mesh_4x1.cpp - ADG test: 4x1 vertical array -------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("topo_mesh_4x1");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(5, 5)
      .setType(Type::i32());

  auto mesh = builder.buildMesh(4, 1, pe, sw, Topology::Mesh);

  // Chain PEs in row-major order: [0][0] -> [1][0] -> [2][0] -> [3][0]
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto e = builder.addModuleInput("e", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(a, mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(b, mesh.peGrid[0][0], 1);
  builder.connectPorts(mesh.peGrid[0][0], 0, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(c, mesh.peGrid[1][0], 1);
  builder.connectPorts(mesh.peGrid[1][0], 0, mesh.peGrid[2][0], 0);
  builder.connectToModuleInput(d, mesh.peGrid[2][0], 1);
  builder.connectPorts(mesh.peGrid[2][0], 0, mesh.peGrid[3][0], 0);
  builder.connectToModuleInput(e, mesh.peGrid[3][0], 1);
  builder.connectToModuleOutput(mesh.peGrid[3][0], 0, out);

  builder.exportMLIR("Output/topo_mesh_4x1.fabric.mlir");
  return 0;
}
