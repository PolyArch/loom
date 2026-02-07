//===-- mixed_asymmetric_switch_mesh.cpp - ADG test: 2x2 mesh asymmetric switch -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_asymmetric_switch_mesh");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  // Asymmetric switch: 5 in, 5 out (min for mesh) but we use it in mesh
  auto sw = builder.newSwitch("asym_xbar")
      .setPortCount(5, 5)
      .setType(Type::i32());

  auto mesh = builder.buildMesh(2, 2, pe, sw, Topology::Mesh);

  // Also add a standalone asymmetric switch for custom connectivity
  auto asw = builder.newSwitch("extra_asym")
      .setPortCount(4, 3)
      .setType(Type::i32());

  auto asw0 = builder.clone(asw, "asw0");

  // Module I/O
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto e = builder.addModuleInput("e", Type::i32());
  auto mesh_out = builder.addModuleOutput("mesh_result", Type::i32());
  auto asw_out0 = builder.addModuleOutput("asw_out0", Type::i32());
  auto asw_out1 = builder.addModuleOutput("asw_out1", Type::i32());

  // Chain mesh PEs: [0][0] -> [0][1] -> [1][0] -> [1][1]
  builder.connectToModuleInput(a, mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(b, mesh.peGrid[0][0], 1);
  builder.connectPorts(mesh.peGrid[0][0], 0, mesh.peGrid[0][1], 0);
  builder.connectToModuleInput(c, mesh.peGrid[0][1], 1);
  builder.connectPorts(mesh.peGrid[0][1], 0, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(d, mesh.peGrid[1][0], 1);
  builder.connectPorts(mesh.peGrid[1][0], 0, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(e, mesh.peGrid[1][1], 1);
  builder.connectToModuleOutput(mesh.peGrid[1][1], 0, mesh_out);

  // Asymmetric switch: 4 inputs (a, b, c, mesh output), 3 outputs
  builder.connectToModuleInput(a, asw0, 0);
  builder.connectToModuleInput(b, asw0, 1);
  builder.connectToModuleInput(c, asw0, 2);
  builder.connectPorts(mesh.peGrid[1][1], 0, asw0, 3);
  builder.connectToModuleOutput(asw0, 0, asw_out0);
  builder.connectToModuleOutput(asw0, 1, asw_out1);
  auto asw_out2 = builder.addModuleOutput("asw_out2", Type::i32());
  builder.connectToModuleOutput(asw0, 2, asw_out2);

  builder.exportMLIR("Output/mixed_asymmetric_switch_mesh.fabric.mlir");
  return 0;
}
