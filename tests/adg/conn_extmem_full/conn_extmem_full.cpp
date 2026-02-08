//===-- conn_extmem_full.cpp - ADG test: extmem load+store pipeline -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_extmem_full");

  auto emem = builder.newExtMemory("dram_ld_st")
      .setLoadPorts(1)
      .setStorePorts(1)
      .setQueueDepth(1)
      .setShape(MemrefType::dynamic1D(Type::i32()));

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto em0 = builder.clone(emem, "em0");
  auto add0 = builder.clone(adder, "add0");

  // ExtMemory first input is memref
  auto mref = builder.addModuleInput("mem", MemrefType::dynamic1D(Type::i32()));
  auto ld_addr = builder.addModuleInput("ld_addr", Type::index());
  auto st_addr = builder.addModuleInput("st_addr", Type::index());
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto lddata = builder.addModuleOutput("lddata", Type::i32());
  auto lddone = builder.addModuleOutput("lddone", Type::none());
  auto stdone = builder.addModuleOutput("stdone", Type::none());

  // Adder: a + b -> store data (no cycle: adder reads from module inputs)
  builder.connectToModuleInput(a, add0, 0);
  builder.connectToModuleInput(b, add0, 1);

  // ExtMemory inputs: [memref, ld_addr, st_addr, st_data]
  builder.connectToModuleInput(mref, em0, 0);
  builder.connectToModuleInput(ld_addr, em0, 1);
  builder.connectToModuleInput(st_addr, em0, 2);
  builder.connectPorts(add0, 0, em0, 3); // adder -> st_data

  // ExtMemory outputs: [lddata, lddone, stdone]
  builder.connectToModuleOutput(em0, 0, lddata);
  builder.connectToModuleOutput(em0, 1, lddone);
  builder.connectToModuleOutput(em0, 2, stdone);

  builder.exportMLIR("Output/conn_extmem_full.fabric.mlir");
  return 0;
}
