//===-- conn_memory_store.cpp - ADG test: addr+data PE -> mem store *- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_memory_store");

  auto mem = builder.newMemory("sram")
      .setLoadPorts(0)
      .setStorePorts(1)
      .setQueueDepth(1)
      .setShape(MemrefType::static1D(64, Type::i32()));

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto add0 = builder.clone(adder, "add0");
  auto mem0 = builder.clone(mem, "mem0");

  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto addr = builder.addModuleInput("addr", Type::index());
  auto done = builder.addModuleOutput("done", Type::none());

  builder.connectToModuleInput(a, add0, 0);
  builder.connectToModuleInput(b, add0, 1);
  // Memory inputs: [st_addr, st_data]
  builder.connectToModuleInput(addr, mem0, 0);
  builder.connectPorts(add0, 0, mem0, 1);
  // Memory outputs (private, 0 ld, 1 st): [lddone(0), stdone(1)]
  builder.connectToModuleOutput(mem0, 0, done);    // lddone
  auto stdone = builder.addModuleOutput("stdone", Type::none());
  builder.connectToModuleOutput(mem0, 1, stdone);  // stdone

  builder.exportMLIR("Output/conn_memory_store.fabric.mlir");
  return 0;
}
