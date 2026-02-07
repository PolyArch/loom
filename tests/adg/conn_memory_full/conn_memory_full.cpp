//===-- conn_memory_full.cpp - ADG test: full load+store pipeline -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_memory_full");

  auto mem = builder.newMemory("sram")
      .setLoadPorts(1)
      .setStorePorts(1)
      .setQueueDepth(4)
      .setShape(MemrefType::static1D(128, Type::i32()));

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  // Separate load and store: addr produces data, adder writes to store.
  // No cycle: adder inputs come from module inputs, not from memory.
  auto add0 = builder.clone(adder, "add0");
  auto mem0 = builder.clone(mem, "mem0");

  auto ld_addr = builder.addModuleInput("ld_addr", Type::index());
  auto st_addr = builder.addModuleInput("st_addr", Type::index());
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto lddata = builder.addModuleOutput("lddata", Type::i32());
  auto done = builder.addModuleOutput("done", Type::none());

  // Adder: a + b -> store data
  builder.connectToModuleInput(a, add0, 0);
  builder.connectToModuleInput(b, add0, 1);
  // Memory inputs: [ld_addr, st_addr, st_data]
  builder.connectToModuleInput(ld_addr, mem0, 0);
  builder.connectToModuleInput(st_addr, mem0, 1);
  builder.connectPorts(add0, 0, mem0, 2); // adder -> st_data
  // Memory outputs: [lddata, lddone, stdone]
  builder.connectToModuleOutput(mem0, 0, lddata);
  auto lddone = builder.addModuleOutput("lddone", Type::none());
  builder.connectToModuleOutput(mem0, 1, lddone); // lddone
  builder.connectToModuleOutput(mem0, 2, done);    // stdone

  builder.exportMLIR("Output/conn_memory_full.fabric.mlir");
  return 0;
}
