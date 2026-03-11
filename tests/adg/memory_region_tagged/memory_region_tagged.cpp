//===-- memory_region_tagged.cpp - ADG test: multi-port mem with regions -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("memory_region_tagged");

  // TAG_WIDTH = clog2(max(4,2)) = 2
  Type tagType = Type::iN(2);

  auto mem = builder.newMemory("spad_region")
      .setLoadPorts(4)
      .setStorePorts(2)
      .setQueueDepth(4)
      .setNumRegion(3)
      .setShape(MemrefType::static1D(1024, Type::i32()));

  auto inst = builder.clone(mem, "mem0");

  // Single tagged port per category
  auto st_data = builder.addModuleInput(
      "st_data", Type::tagged(Type::i32(), tagType));
  auto st_addr = builder.addModuleInput(
      "st_addr", Type::tagged(Type::index(), tagType));
  auto ld_addr = builder.addModuleInput(
      "ld_addr", Type::tagged(Type::index(), tagType));

  auto ld_data = builder.addModuleOutput(
      "ld_data", Type::tagged(Type::i32(), tagType));
  auto ld_done = builder.addModuleOutput(
      "ld_done", Type::tagged(Type::none(), tagType));
  auto st_done = builder.addModuleOutput(
      "st_done", Type::tagged(Type::none(), tagType));

  // Memory inputs (ld=4, st=2): [st_data(0), st_addr(1), ld_addr(2)]
  builder.connectToModuleInput(st_data, inst, 0);
  builder.connectToModuleInput(st_addr, inst, 1);
  builder.connectToModuleInput(ld_addr, inst, 2);

  // Memory outputs: [ld_data(0), ld_done(1), st_done(2)]
  builder.connectToModuleOutput(inst, 0, ld_data);
  builder.connectToModuleOutput(inst, 1, ld_done);
  builder.connectToModuleOutput(inst, 2, st_done);

  builder.exportMLIR("Output/memory_region_tagged.fabric.mlir");
  return 0;
}
