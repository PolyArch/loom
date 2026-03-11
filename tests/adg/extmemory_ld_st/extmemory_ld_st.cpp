//===-- extmemory_ld_st.cpp - ADG test: extmem 1-load 1-store -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("extmemory_ld_st");

  auto emem = builder.newExtMemory("dram_ld_st")
      .setLoadPorts(1)
      .setStorePorts(1)
      .setQueueDepth(4)
      .setShape(MemrefType::dynamic1D(Type::i32()));

  auto inst = builder.clone(emem, "em0");

  auto mref = builder.addModuleInput("m", MemrefType::dynamic1D(Type::i32()));
  auto ldaddr = builder.addModuleInput("ldaddr", Type::index());
  auto staddr = builder.addModuleInput("staddr", Type::index());
  auto stdata = builder.addModuleInput("stdata", Type::i32());
  auto lddata = builder.addModuleOutput("lddata", Type::i32());
  auto lddone = builder.addModuleOutput("lddone", Type::none());
  auto stdone = builder.addModuleOutput("stdone", Type::none());

  // ExtMemory inputs (ld=1, st=1): [memref(0), st_data(1), st_addr(2), ld_addr(3)]
  builder.connectToModuleInput(mref, inst, 0);
  builder.connectToModuleInput(stdata, inst, 1);    // st_data_0
  builder.connectToModuleInput(staddr, inst, 2);    // st_addr_0
  builder.connectToModuleInput(ldaddr, inst, 3);    // ld_addr_0
  // ExtMemory outputs (1 ld, 1 st): [ld_data(0), ld_done(1), st_done(2)]
  builder.connectToModuleOutput(inst, 0, lddata);
  builder.connectToModuleOutput(inst, 1, lddone);
  builder.connectToModuleOutput(inst, 2, stdone);

  builder.exportMLIR("Output/extmemory_ld_st.fabric.mlir");
  return 0;
}
