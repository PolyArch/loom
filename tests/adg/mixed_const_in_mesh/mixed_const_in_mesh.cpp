//===-- mixed_const_in_mesh.cpp - ADG test: 2x2 mesh + constant PE -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_const_in_mesh");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(5, 5)
      .setType(Type::i32());

  auto cpe = builder.newConstantPE("const_gen")
      .setOutputType(Type::i32());

  auto mesh = builder.buildMesh(2, 2, pe, sw, Topology::Mesh);
  auto c0 = builder.clone(cpe, "c0");

  // Module I/O
  auto ctrl = builder.addModuleInput("ctrl", Type::none());
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto result = builder.addModuleOutput("result", Type::i32());

  // Constant PE feeds PE[0][0] input 0
  builder.connectToModuleInput(ctrl, c0, 0);
  builder.connectPorts(c0, 0, mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(a, mesh.peGrid[0][0], 1);

  // Chain PEs in row-major order: [0][0] -> [0][1] -> [1][0] -> [1][1]
  builder.connectPorts(mesh.peGrid[0][0], 0, mesh.peGrid[0][1], 0);
  builder.connectToModuleInput(b, mesh.peGrid[0][1], 1);
  builder.connectPorts(mesh.peGrid[0][1], 0, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(c, mesh.peGrid[1][0], 1);
  builder.connectPorts(mesh.peGrid[1][0], 0, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(d, mesh.peGrid[1][1], 1);

  // Output from PE[1][1]
  builder.connectToModuleOutput(mesh.peGrid[1][1], 0, result);

  builder.exportMLIR("Output/mixed_const_in_mesh.fabric.mlir");
  return 0;
}
