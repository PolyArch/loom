//===-- mixed_asymmetric_switch_mesh.cpp - ADG test: 2x2 mesh asymmetric switch -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

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
  auto bcast_0_sw_def = builder.newSwitch("bcast_0_sw")
      .setPortCount(1, 2)
      .setType(Type::i32());
  auto bcast_0 = builder.clone(bcast_0_sw_def, "bcast_0");
  builder.connectToModuleInput(a, bcast_0, 0);
  builder.connectPorts(bcast_0, 0, mesh.peGrid[0][0], 0);
  builder.connectPorts(bcast_0, 1, asw0, 0);
  auto bcast_1_sw_def = builder.newSwitch("bcast_1_sw")
      .setPortCount(1, 2)
      .setType(Type::i32());
  auto bcast_1 = builder.clone(bcast_1_sw_def, "bcast_1");
  builder.connectToModuleInput(b, bcast_1, 0);
  builder.connectPorts(bcast_1, 0, mesh.peGrid[0][0], 1);
  builder.connectPorts(bcast_1, 1, asw0, 1);
  auto chain_in_0 = builder.addModuleInput("chain_in_0", Type::i32());
  builder.connectToModuleInput(chain_in_0, mesh.peGrid[0][1], 0);
  auto bcast_2_sw_def = builder.newSwitch("bcast_2_sw")
      .setPortCount(1, 2)
      .setType(Type::i32());
  auto bcast_2 = builder.clone(bcast_2_sw_def, "bcast_2");
  builder.connectToModuleInput(c, bcast_2, 0);
  builder.connectPorts(bcast_2, 0, mesh.peGrid[0][1], 1);
  builder.connectPorts(bcast_2, 1, asw0, 2);
  auto chain_in_1 = builder.addModuleInput("chain_in_1", Type::i32());
  builder.connectToModuleInput(chain_in_1, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(d, mesh.peGrid[1][0], 1);
  auto chain_in_2 = builder.addModuleInput("chain_in_2", Type::i32());
  builder.connectToModuleInput(chain_in_2, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(e, mesh.peGrid[1][1], 1);
  auto chain_in_3 = builder.addModuleInput("chain_in_3", Type::i32());
  auto chain_in_3_pass_sw = builder.newSwitch("chain_in_3_pass_sw")
      .setPortCount(1, 1)
      .setType(Type::i32());
  auto chain_in_3_pass = builder.clone(chain_in_3_pass_sw, "chain_in_3_pass");
  builder.connectToModuleInput(chain_in_3, chain_in_3_pass, 0);
  builder.connectToModuleOutput(chain_in_3_pass, 0, mesh_out);

  // Asymmetric switch: 4 inputs (a, b, c, mesh output), 3 outputs
  auto chain_in_4 = builder.addModuleInput("chain_in_4", Type::i32());
  builder.connectToModuleInput(chain_in_4, asw0, 3);
  builder.connectToModuleOutput(asw0, 0, asw_out0);
  builder.connectToModuleOutput(asw0, 1, asw_out1);
  auto asw_out2 = builder.addModuleOutput("asw_out2", Type::i32());
  builder.connectToModuleOutput(asw0, 2, asw_out2);

  builder.exportMLIR("Output/mixed_asymmetric_switch_mesh.fabric.mlir");
  return 0;
}
