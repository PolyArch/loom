//===-- mixed_multi_mesh_join.cpp - ADG test: two 2x2 meshes joined -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

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
  auto bcast_0_sw_def = builder.newSwitch("bcast_0_sw")
      .setPortCount(1, 2)
      .setType(Type::i32());
  auto bcast_0 = builder.clone(bcast_0_sw_def, "bcast_0");
  builder.connectToModuleInput(b, bcast_0, 0);
  builder.connectPorts(bcast_0, 0, meshA.peGrid[0][0], 1);
  builder.connectPorts(bcast_0, 1, meshA.peGrid[1][1], 1);
  auto chain_in_0 = builder.addModuleInput("chain_in_0", Type::i32());
  builder.connectToModuleInput(chain_in_0, meshA.peGrid[0][1], 0);
  builder.connectToModuleInput(c, meshA.peGrid[0][1], 1);
  auto chain_in_1 = builder.addModuleInput("chain_in_1", Type::i32());
  builder.connectToModuleInput(chain_in_1, meshA.peGrid[1][0], 0);
  builder.connectToModuleInput(d, meshA.peGrid[1][0], 1);
  auto chain_in_2 = builder.addModuleInput("chain_in_2", Type::i32());
  builder.connectToModuleInput(chain_in_2, meshA.peGrid[1][1], 0);

  // mesh A output -> bridge -> mesh B input
  // Bridge: all 3 inputs connected
  auto chain_in_3 = builder.addModuleInput("chain_in_3", Type::i32());
  builder.connectToModuleInput(chain_in_3, br0, 0);
  builder.connectToModuleInput(e, br0, 1);
  builder.connectToModuleInput(br_in, br0, 2);

  // Chain mesh B PEs: [0][0] -> [0][1] -> [1][0] -> [1][1]
  builder.connectPorts(br0, 0, meshB.peGrid[0][0], 0);
  auto bcast_1_sw_def = builder.newSwitch("bcast_1_sw")
      .setPortCount(1, 2)
      .setType(Type::i32());
  auto bcast_1 = builder.clone(bcast_1_sw_def, "bcast_1");
  builder.connectToModuleInput(f, bcast_1, 0);
  builder.connectPorts(bcast_1, 0, meshB.peGrid[0][0], 1);
  builder.connectPorts(bcast_1, 1, meshB.peGrid[1][1], 1);
  auto chain_in_4 = builder.addModuleInput("chain_in_4", Type::i32());
  builder.connectToModuleInput(chain_in_4, meshB.peGrid[0][1], 0);
  builder.connectToModuleInput(g, meshB.peGrid[0][1], 1);
  auto chain_in_5 = builder.addModuleInput("chain_in_5", Type::i32());
  builder.connectToModuleInput(chain_in_5, meshB.peGrid[1][0], 0);
  builder.connectToModuleInput(h, meshB.peGrid[1][0], 1);
  auto chain_in_6 = builder.addModuleInput("chain_in_6", Type::i32());
  builder.connectToModuleInput(chain_in_6, meshB.peGrid[1][1], 0);

  // Bridge remaining outputs -> module outputs
  auto br_out1 = builder.addModuleOutput("br_out1", Type::i32());
  auto br_out2 = builder.addModuleOutput("br_out2", Type::i32());
  builder.connectToModuleOutput(br0, 1, br_out1);
  builder.connectToModuleOutput(br0, 2, br_out2);

  // mesh B output -> result
  auto chain_in_7 = builder.addModuleInput("chain_in_7", Type::i32());
  auto chain_in_7_pass_sw = builder.newSwitch("chain_in_7_pass_sw")
      .setPortCount(1, 1)
      .setType(Type::i32());
  auto chain_in_7_pass = builder.clone(chain_in_7_pass_sw, "chain_in_7_pass");
  builder.connectToModuleInput(chain_in_7, chain_in_7_pass, 0);
  builder.connectToModuleOutput(chain_in_7_pass, 0, result);

  builder.exportMLIR("Output/mixed_multi_mesh_join.fabric.mlir");
  return 0;
}
