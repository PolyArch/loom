//===-- topo_mesh_multi_pe.cpp - ADG test: mesh with extra PE instances -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("topo_mesh_multi_pe");

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto mul = builder.newPE("mul")
      .setLatency(2, 2, 2)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.muli");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(5, 5)
      .setType(Type::i32());

  // Build the base 2x2 mesh with adder PEs
  auto mesh = builder.buildMesh(2, 2, adder, sw, Topology::Mesh);

  // Clone extra multiplier PEs
  auto mul0 = builder.clone(mul, "mul_extra_0");
  auto mul1 = builder.clone(mul, "mul_extra_1");

  // Module I/O
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto e = builder.addModuleInput("e", Type::i32());
  auto f = builder.addModuleInput("f", Type::i32());
  auto g = builder.addModuleInput("g", Type::i32());
  auto outMul = builder.addModuleOutput("result_mul", Type::i32());

  // Chain mesh PEs: [0][0] -> [0][1] -> [1][0] -> [1][1]
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

  // Feed extra multiplier PEs
  builder.connectToModuleInput(f, mul0, 0);
  builder.connectToModuleInput(g, mul0, 1);

  // Chain: mul0 -> mul1 input 0, mesh[1][1] output -> mul1 input 1
  builder.connectPorts(mul0, 0, mul1, 0);
  auto bcast_3 = builder.addModuleInput("bcast_3", Type::i32());
  builder.connectToModuleInput(bcast_3, mul1, 1);

  // Outputs
  builder.connectToModuleOutput(mul1, 0, outMul);

  builder.exportMLIR("Output/topo_mesh_multi_pe.fabric.mlir");
  return 0;
}
