//===-- memory_ld_st.cpp - ADG test: 1-load 1-store memory ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("memory_ld_st");

  auto mem = builder.newMemory("spad_64")
      .setLoadPorts(1)
      .setStorePorts(1)
      .setQueueDepth(4)
      .setShape(MemrefType::static1D(64, Type::i32()));

  auto inst = builder.clone(mem, "mem0");

  // Inputs: ldaddr, staddr, stdata
  auto ldaddr = builder.addModuleInput("ldaddr", Type::index());
  auto staddr = builder.addModuleInput("staddr", Type::index());
  auto stdata = builder.addModuleInput("stdata", Type::i32());
  // Outputs: lddata, lddone, stdone
  auto lddata = builder.addModuleOutput("lddata", Type::i32());
  auto lddone = builder.addModuleOutput("lddone", Type::none());
  auto stdone = builder.addModuleOutput("stdone", Type::none());

  builder.connectToModuleInput(ldaddr, inst, 0);
  builder.connectToModuleInput(staddr, inst, 1);
  builder.connectToModuleInput(stdata, inst, 2);
  builder.connectToModuleOutput(inst, 0, lddata);
  builder.connectToModuleOutput(inst, 1, lddone);
  builder.connectToModuleOutput(inst, 2, stdone);

  builder.exportMLIR("Output/memory_ld_st.fabric.mlir");
  return 0;
}
