//===-- mixed_all_memory_types.cpp - ADG test: memory + extmemory -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_all_memory_types");

  auto mem = builder.newMemory("sram")
      .setLoadPorts(1)
      .setStorePorts(0)
      .setShape(MemrefType::static1D(128, Type::i32()));

  auto emem = builder.newExtMemory("dram")
      .setLoadPorts(1)
      .setStorePorts(0)
      .setShape(MemrefType::dynamic1D(Type::i32()));

  auto mem0 = builder.clone(mem, "sram0");
  auto em0 = builder.clone(emem, "dram0");

  // Module I/O for private memory
  auto ld_addr = builder.addModuleInput("ld_addr", Type::index());
  auto lddata = builder.addModuleOutput("lddata", Type::i32());
  auto lddone = builder.addModuleOutput("lddone", Type::none());

  // Module I/O for external memory
  auto mref = builder.addModuleInput("mem", MemrefType::dynamic1D(Type::i32()));
  auto ext_addr = builder.addModuleInput("ext_addr", Type::index());
  auto ext_data = builder.addModuleOutput("ext_data", Type::i32());
  auto ext_done = builder.addModuleOutput("ext_done", Type::none());

  // Private memory: ld_addr -> lddata, lddone
  builder.connectToModuleInput(ld_addr, mem0, 0);
  builder.connectToModuleOutput(mem0, 0, lddata);
  builder.connectToModuleOutput(mem0, 1, lddone);

  // External memory: memref, ld_addr -> lddata, lddone
  builder.connectToModuleInput(mref, em0, 0);
  builder.connectToModuleInput(ext_addr, em0, 1);
  builder.connectToModuleOutput(em0, 0, ext_data);
  builder.connectToModuleOutput(em0, 1, ext_done);

  builder.exportMLIR("Output/mixed_all_memory_types.fabric.mlir");
  return 0;
}
