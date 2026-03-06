//===-- memory_region_tagged.cpp - ADG test: tagged mem with regions -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("memory_region_tagged");

  auto tagType = Type::iN(2);
  auto taggedIndex = Type::tagged(Type::index(), tagType);
  auto taggedData = Type::tagged(Type::i32(), tagType);
  auto taggedNone = Type::tagged(Type::none(), tagType);

  auto mem = builder.newMemory("spad_tagged_region")
      .setLoadPorts(4)
      .setStorePorts(2)
      .setQueueDepth(4)
      .setNumRegion(3)
      .setShape(MemrefType::static1D(1024, Type::i32()));

  auto inst = builder.clone(mem, "mem0");

  auto ld_addr = builder.addModuleInput("ld_addr", taggedIndex);
  auto st_addr = builder.addModuleInput("st_addr", taggedIndex);
  auto st_data = builder.addModuleInput("st_data", taggedData);
  auto ld_data = builder.addModuleOutput("ld_data", taggedData);
  auto ld_done = builder.addModuleOutput("ld_done", taggedNone);
  auto st_done = builder.addModuleOutput("st_done", taggedNone);

  builder.connectToModuleInput(ld_addr, inst, 0);
  builder.connectToModuleInput(st_addr, inst, 1);
  builder.connectToModuleInput(st_data, inst, 2);
  builder.connectToModuleOutput(inst, 0, ld_data);
  builder.connectToModuleOutput(inst, 1, ld_done);
  builder.connectToModuleOutput(inst, 2, st_done);

  builder.exportMLIR("Output/memory_region_tagged.fabric.mlir");
  return 0;
}
