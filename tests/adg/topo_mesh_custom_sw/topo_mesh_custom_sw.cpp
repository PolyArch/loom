//===-- topo_mesh_custom_sw.cpp - ADG test: mesh with custom switch -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("topo_mesh_custom_sw");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  // Custom switch with partial connectivity (not full crossbar).
  // 5 inputs (N=0, E=1, S=2, W=3, PE=4), 5 outputs.
  // Output-major connectivity table: table[out][in].
  // Allow only: N->S, S->N, E->W, W->E, PE->all, all->PE.
  auto sw = builder.newSwitch("partial_xbar")
      .setPortCount(5, 5)
      .setType(Type::i32())
      .setConnectivity({
        // out0 (N): can receive from S(2), PE(4)
        {false, false, true,  false, true},
        // out1 (E): can receive from W(3), PE(4)
        {false, false, false, true,  true},
        // out2 (S): can receive from N(0), PE(4)
        {true,  false, false, false, true},
        // out3 (W): can receive from E(1), PE(4)
        {false, true,  false, false, true},
        // out4 (PE): can receive from all
        {true,  true,  true,  true,  false}
      });

  auto mesh = builder.buildMesh(2, 2, pe, sw, Topology::Mesh);

  // Chain PEs in row-major order: [0][0] -> [0][1] -> [1][0] -> [1][1]
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto e = builder.addModuleInput("e", Type::i32());

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

  builder.exportMLIR("Output/topo_mesh_custom_sw.fabric.mlir");
  return 0;
}
