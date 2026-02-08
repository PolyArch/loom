//===-- extmemory_single_ld.cpp - ADG test: extmem 1-load -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("extmemory_single_ld");

  auto emem = builder.newExtMemory("dram_ld")
      .setLoadPorts(1)
      .setStorePorts(0)
      .setShape(MemrefType::dynamic1D(Type::i32()));

  auto inst = builder.clone(emem, "em0");

  auto mref = builder.addModuleInput("m", MemrefType::dynamic1D(Type::i32()));
  auto ldaddr = builder.addModuleInput("ldaddr", Type::index());
  auto lddata = builder.addModuleOutput("lddata", Type::i32());
  auto lddone = builder.addModuleOutput("lddone", Type::none());

  builder.connectToModuleInput(mref, inst, 0);
  builder.connectToModuleInput(ldaddr, inst, 1);
  builder.connectToModuleOutput(inst, 0, lddata);
  builder.connectToModuleOutput(inst, 1, lddone);

  builder.exportMLIR("Output/extmemory_single_ld.fabric.mlir");
  return 0;
}
