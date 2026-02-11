//===-- topo_mesh_with_io.cpp - ADG test: mesh with multiple I/Os -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("topo_mesh_with_io");

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

  // Extra module I/O beyond what's needed for the chain
  auto extra0 = builder.addModuleInput("extra0", Type::i32());
  auto extra1 = builder.addModuleInput("extra1", Type::i32());
  auto extra2 = builder.addModuleInput("extra2", Type::i32());

  // Module outputs from each corner PE and chain output

  builder.connectToModuleInput(a, mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(b, mesh.peGrid[0][0], 1);
  auto bcast_0 = builder.addModuleInput("bcast_0", Type::i32());
  builder.connectToModuleInput(bcast_0, mesh.peGrid[0][1], 0);
  builder.connectToModuleInput(c, mesh.peGrid[0][1], 1);
  auto bcast_1 = builder.addModuleInput("bcast_1", Type::i32());
  builder.connectToModuleInput(bcast_1, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(d, mesh.peGrid[1][0], 1);
  auto bcast_2 = builder.addModuleInput("bcast_2", Type::i32());
  builder.connectToModuleInput(bcast_2, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(e, mesh.peGrid[1][1], 1);

  // Connect outputs from all four PEs

  builder.exportMLIR("Output/topo_mesh_with_io.fabric.mlir");
  return 0;
}
