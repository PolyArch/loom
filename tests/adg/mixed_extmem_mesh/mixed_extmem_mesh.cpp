//===-- mixed_extmem_mesh.cpp - ADG test: 2x2 mesh + extmem -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_extmem_mesh");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(5, 5)
      .setType(Type::i32());

  auto emem = builder.newExtMemory("dram")
      .setLoadPorts(1)
      .setStorePorts(0)
      .setShape(MemrefType::dynamic1D(Type::i32()));

  auto mesh = builder.buildMesh(2, 2, pe, sw, Topology::Mesh);
  auto em0 = builder.clone(emem, "em0");

  // Module I/O
  auto mref = builder.addModuleInput("mem", MemrefType::dynamic1D(Type::i32()));
  auto addr = builder.addModuleInput("addr", Type::index());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto result = builder.addModuleOutput("result", Type::i32());
  auto done = builder.addModuleOutput("done", Type::none());

  // ExtMemory: memref + ld_addr -> lddata
  builder.connectToModuleInput(mref, em0, 0);
  builder.connectToModuleInput(addr, em0, 1);

  // ExtMemory lddata -> corner PE[0][0] input 0
  builder.connectPorts(em0, 0, mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(b, mesh.peGrid[0][0], 1);

  // Chain PEs in row-major order: [0][0] -> [0][1] -> [1][0] -> [1][1]
  builder.connectPorts(mesh.peGrid[0][0], 0, mesh.peGrid[0][1], 0);
  builder.connectToModuleInput(c, mesh.peGrid[0][1], 1);
  builder.connectPorts(mesh.peGrid[0][1], 0, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(d, mesh.peGrid[1][0], 1);
  builder.connectPorts(mesh.peGrid[1][0], 0, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(b, mesh.peGrid[1][1], 1);

  // Output from PE[1][1]
  builder.connectToModuleOutput(mesh.peGrid[1][1], 0, result);
  builder.connectToModuleOutput(em0, 1, done);

  builder.exportMLIR("Output/mixed_extmem_mesh.fabric.mlir");
  return 0;
}
