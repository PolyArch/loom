//===-- mixed_multi_mesh_join.cpp - ADG test: two 2x2 meshes joined -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_multi_mesh_join");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(5, 5)
      .setType(Type::i32());

  // Bridge switch connecting the two meshes
  auto bridge = builder.newSwitch("bridge")
      .setPortCount(3, 3)
      .setType(Type::i32());

  auto meshA = builder.buildMesh(2, 2, pe, sw, Topology::Mesh);
  auto meshB = builder.buildMesh(2, 2, pe, sw, Topology::Mesh);
  auto br0 = builder.clone(bridge, "br0");

  // Module I/O
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto e = builder.addModuleInput("e", Type::i32());
  auto f = builder.addModuleInput("f", Type::i32());
  auto g = builder.addModuleInput("g", Type::i32());
  auto h = builder.addModuleInput("h", Type::i32());
  auto br_in = builder.addModuleInput("br_in", Type::i32());
  auto result = builder.addModuleOutput("result", Type::i32());

  // Chain mesh A PEs: [0][0] -> [0][1] -> [1][0] -> [1][1]
  builder.connectToModuleInput(a, meshA.peGrid[0][0], 0);
  builder.connectToModuleInput(b, meshA.peGrid[0][0], 1);
  builder.connectPorts(meshA.peGrid[0][0], 0, meshA.peGrid[0][1], 0);
  builder.connectToModuleInput(c, meshA.peGrid[0][1], 1);
  builder.connectPorts(meshA.peGrid[0][1], 0, meshA.peGrid[1][0], 0);
  builder.connectToModuleInput(d, meshA.peGrid[1][0], 1);
  builder.connectPorts(meshA.peGrid[1][0], 0, meshA.peGrid[1][1], 0);
  builder.connectToModuleInput(b, meshA.peGrid[1][1], 1);

  // mesh A output -> bridge -> mesh B input
  // Bridge: all 3 inputs connected
  builder.connectPorts(meshA.peGrid[1][1], 0, br0, 0);
  builder.connectToModuleInput(e, br0, 1);
  builder.connectToModuleInput(br_in, br0, 2);

  // Chain mesh B PEs: [0][0] -> [0][1] -> [1][0] -> [1][1]
  builder.connectPorts(br0, 0, meshB.peGrid[0][0], 0);
  builder.connectToModuleInput(f, meshB.peGrid[0][0], 1);
  builder.connectPorts(meshB.peGrid[0][0], 0, meshB.peGrid[0][1], 0);
  builder.connectToModuleInput(g, meshB.peGrid[0][1], 1);
  builder.connectPorts(meshB.peGrid[0][1], 0, meshB.peGrid[1][0], 0);
  builder.connectToModuleInput(h, meshB.peGrid[1][0], 1);
  builder.connectPorts(meshB.peGrid[1][0], 0, meshB.peGrid[1][1], 0);
  builder.connectToModuleInput(f, meshB.peGrid[1][1], 1);

  // Bridge remaining outputs -> module outputs
  auto br_out1 = builder.addModuleOutput("br_out1", Type::i32());
  auto br_out2 = builder.addModuleOutput("br_out2", Type::i32());
  builder.connectToModuleOutput(br0, 1, br_out1);
  builder.connectToModuleOutput(br0, 2, br_out2);

  // mesh B output -> result
  builder.connectToModuleOutput(meshB.peGrid[1][1], 0, result);

  builder.exportMLIR("Output/mixed_multi_mesh_join.fabric.mlir");
  return 0;
}
