//===-- mixed_torus_with_mem.cpp - ADG test: 2x2 torus + memory -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_torus_with_mem");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(5, 5)
      .setType(Type::i32());

  auto mem = builder.newMemory("sram")
      .setLoadPorts(1)
      .setStorePorts(1)
      .setQueueDepth(4)
      .setShape(MemrefType::static1D(128, Type::i32()));

  auto mesh = builder.buildMesh(2, 2, pe, sw, Topology::Torus);
  auto mem0 = builder.clone(mem, "mem0");

  // Module I/O
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto ld_addr = builder.addModuleInput("ld_addr", Type::index());
  auto st_addr = builder.addModuleInput("st_addr", Type::index());
  auto mesh_out = builder.addModuleOutput("mesh_result", Type::i32());
  auto lddata = builder.addModuleOutput("lddata", Type::i32());
  auto stdone = builder.addModuleOutput("stdone", Type::none());

  // Chain PEs in row-major order: [0][0] -> [0][1] -> [1][0] -> [1][1]
  builder.connectToModuleInput(a, mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(b, mesh.peGrid[0][0], 1);
  builder.connectPorts(mesh.peGrid[0][0], 0, mesh.peGrid[0][1], 0);
  builder.connectToModuleInput(c, mesh.peGrid[0][1], 1);
  builder.connectPorts(mesh.peGrid[0][1], 0, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(d, mesh.peGrid[1][0], 1);
  builder.connectPorts(mesh.peGrid[1][0], 0, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(b, mesh.peGrid[1][1], 1);
  builder.connectToModuleOutput(mesh.peGrid[1][1], 0, mesh_out);

  // Memory connected to corner PE[0][0] output as store data
  // Memory inputs: [ld_addr, st_addr, st_data]
  builder.connectToModuleInput(ld_addr, mem0, 0);
  builder.connectToModuleInput(st_addr, mem0, 1);
  builder.connectPorts(mesh.peGrid[0][0], 0, mem0, 2); // mesh PE output -> st_data
  // Memory outputs: [lddata, lddone, stdone]
  builder.connectToModuleOutput(mem0, 0, lddata);
  builder.connectToModuleOutput(mem0, 2, stdone);

  builder.exportMLIR("Output/mixed_torus_with_mem.fabric.mlir");
  return 0;
}
