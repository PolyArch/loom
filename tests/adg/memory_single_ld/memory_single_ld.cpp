//===-- memory_single_ld.cpp - ADG test: 1-load memory --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("memory_single_ld");

  auto mem = builder.newMemory("rom_64")
      .setLoadPorts(1)
      .setStorePorts(0)
      .setShape(MemrefType::static1D(64, Type::i32()));

  auto inst = builder.clone(mem, "mem0");

  auto ldaddr = builder.addModuleInput("ldaddr", Type::index());
  auto lddata = builder.addModuleOutput("lddata", Type::i32());
  auto lddone = builder.addModuleOutput("lddone", Type::none());

  builder.connectToModuleInput(ldaddr, inst, 0);
  builder.connectToModuleOutput(inst, 0, lddata);
  builder.connectToModuleOutput(inst, 1, lddone);

  builder.exportMLIR("Output/memory_single_ld.fabric.mlir");
  return 0;
}
