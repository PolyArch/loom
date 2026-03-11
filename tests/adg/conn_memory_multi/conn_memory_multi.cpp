//===-- conn_memory_multi.cpp - ADG test: multi-port memory ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_memory_multi");

  // TAG_WIDTH = clog2(max(2,2)) = 1
  Type tagType = Type::iN(1);

  // 2 load + 2 store multi-port memory (single tagged port per category)
  auto mem = builder.newMemory("spad_multi")
      .setLoadPorts(2)
      .setStorePorts(2)
      .setQueueDepth(4)
      .setShape(MemrefType::static1D(64, Type::i32()));

  auto mem0 = builder.clone(mem, "mem0");

  // Module inputs (tagged to match memory port types)
  auto st_data = builder.addModuleInput(
      "st_data", Type::tagged(Type::i32(), tagType));
  auto st_addr = builder.addModuleInput(
      "st_addr", Type::tagged(Type::index(), tagType));
  auto ld_addr = builder.addModuleInput(
      "ld_addr", Type::tagged(Type::index(), tagType));

  // Module outputs
  auto lddata = builder.addModuleOutput(
      "lddata", Type::tagged(Type::i32(), tagType));
  auto lddone = builder.addModuleOutput(
      "lddone", Type::tagged(Type::none(), tagType));
  auto stdone = builder.addModuleOutput(
      "stdone", Type::tagged(Type::none(), tagType));

  // Memory inputs: [st_data(0), st_addr(1), ld_addr(2)]
  builder.connectToModuleInput(st_data, mem0, 0);
  builder.connectToModuleInput(st_addr, mem0, 1);
  builder.connectToModuleInput(ld_addr, mem0, 2);

  // Memory outputs: [ld_data(0), ld_done(1), st_done(2)]
  builder.connectToModuleOutput(mem0, 0, lddata);
  builder.connectToModuleOutput(mem0, 1, lddone);
  builder.connectToModuleOutput(mem0, 2, stdone);

  builder.exportMLIR("Output/conn_memory_multi.fabric.mlir");
  return 0;
}
