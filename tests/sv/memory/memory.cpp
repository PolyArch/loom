//===-- memory.cpp - SV test: memory module --------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("memory");

  auto mem = builder.newMemory("spm")
      .setLoadPorts(1)
      .setStorePorts(1)
      .setQueueDepth(4)
      .setPrivate(true)
      .setShape(MemrefType::static1D(64, Type::i32()));

  auto m0 = builder.clone(mem, "m0");

  // Memory inputs: ldaddr, staddr, stdata
  auto ldaddr = builder.addModuleInput("ldaddr", Type::index());
  auto staddr = builder.addModuleInput("staddr", Type::index());
  auto stdata = builder.addModuleInput("stdata", Type::i32());

  // Memory outputs: lddata (elem type), lddone (none), stdone (none)
  auto lddata = builder.addModuleOutput("lddata", Type::i32());
  auto lddone = builder.addModuleOutput("lddone", Type::none());
  auto stdone = builder.addModuleOutput("stdone", Type::none());

  builder.connectToModuleInput(ldaddr, m0, 0);
  builder.connectToModuleInput(staddr, m0, 1);
  builder.connectToModuleInput(stdata, m0, 2);
  builder.connectToModuleOutput(m0, 0, lddata);
  builder.connectToModuleOutput(m0, 1, lddone);
  builder.connectToModuleOutput(m0, 2, stdone);

  builder.exportMLIR("Output/memory.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
