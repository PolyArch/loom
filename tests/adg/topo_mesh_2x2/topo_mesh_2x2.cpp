//===-- topo_mesh_2x2.cpp - ADG test: 2x2 mesh topology --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("topo_mesh_2x2");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(5, 5)
      .setType(Type::i32());

  auto mesh = builder.buildMesh(2, 2, pe, sw, Topology::Mesh);

  // Chain PEs in row-major order: [0][0] -> [0][1] -> [1][0] -> [1][1]
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto e = builder.addModuleInput("e", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(a, mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(b, mesh.peGrid[0][0], 1);
  builder.connectPorts(mesh.peGrid[0][0], 0, mesh.peGrid[0][1], 0);
  builder.connectToModuleInput(c, mesh.peGrid[0][1], 1);
  builder.connectPorts(mesh.peGrid[0][1], 0, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(d, mesh.peGrid[1][0], 1);
  builder.connectPorts(mesh.peGrid[1][0], 0, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(e, mesh.peGrid[1][1], 1);
  builder.connectToModuleOutput(mesh.peGrid[1][1], 0, out);

  builder.exportMLIR("Output/topo_mesh_2x2.fabric.mlir");
  return 0;
}
