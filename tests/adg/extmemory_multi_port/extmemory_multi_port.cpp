//===-- extmemory_multi_port.cpp - ADG test: multi-port extmem -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("extmemory_multi_port");

  auto emem = builder.newExtMemory("dram_multi")
      .setLoadPorts(2)
      .setStorePorts(0)
      .setShape(MemrefType::dynamic1D(Type::i32()));

  auto inst = builder.clone(emem, "em0");

  // Single tagged port per category (tags embedded in data width)
  Type tagType = Type::iN(1);
  auto mref = builder.addModuleInput("m", MemrefType::dynamic1D(Type::i32()));
  auto ld_addr = builder.addModuleInput(
      "ld_addr", Type::tagged(Type::index(), tagType));
  auto ld_data = builder.addModuleOutput(
      "ld_data", Type::tagged(Type::i32(), tagType));
  auto ld_done = builder.addModuleOutput(
      "ld_done", Type::tagged(Type::none(), tagType));

  // ExtMemory inputs (ld=2, st=0): [memref(0), ld_addr(1)]
  builder.connectToModuleInput(mref, inst, 0);
  builder.connectToModuleInput(ld_addr, inst, 1);
  // ExtMemory outputs: [ld_data(0), ld_done(1)]
  builder.connectToModuleOutput(inst, 0, ld_data);
  builder.connectToModuleOutput(inst, 1, ld_done);

  builder.exportMLIR("Output/extmemory_multi_port.fabric.mlir");
  return 0;
}
