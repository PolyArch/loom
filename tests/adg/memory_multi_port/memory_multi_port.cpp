//===-- memory_multi_port.cpp - ADG test: multi-port memory ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("memory_multi_port");

  auto mem = builder.newMemory("spad_multi_ld")
      .setLoadPorts(2)
      .setStorePorts(0)
      .setShape(MemrefType::static1D(64, Type::i32()));

  auto inst = builder.clone(mem, "mem0");

  // Single tagged port per category (tags embedded in data width)
  Type tagType = Type::iN(1);
  auto ld_addr = builder.addModuleInput(
      "ld_addr", Type::tagged(Type::index(), tagType));
  auto ld_data = builder.addModuleOutput(
      "ld_data", Type::tagged(Type::i32(), tagType));
  auto ld_done = builder.addModuleOutput(
      "ld_done", Type::tagged(Type::none(), tagType));

  // Memory inputs (ld=2, st=0): [ld_addr(0)]
  builder.connectToModuleInput(ld_addr, inst, 0);
  // Memory outputs: [ld_data(0), ld_done(1)]
  builder.connectToModuleOutput(inst, 0, ld_data);
  builder.connectToModuleOutput(inst, 1, ld_done);

  builder.exportMLIR("Output/memory_multi_port.fabric.mlir");
  return 0;
}
