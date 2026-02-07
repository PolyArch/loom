//===-- mixed_memory_network.cpp - ADG test: memory with PE network -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_memory_network");

  auto addrGen = builder.newPE("addr_gen")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::index(), Type::index()})
      .setOutputPorts({Type::index()})
      .addOp("arith.addi");

  auto compute = builder.newPE("compute")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto mem = builder.newMemory("sram")
      .setLoadPorts(1)
      .setStorePorts(0)
      .setShape(MemrefType::static1D(128, Type::i32()));

  auto ag0 = builder.clone(addrGen, "ag0");
  auto mem0 = builder.clone(mem, "mem0");
  auto comp0 = builder.clone(compute, "comp0");

  // Module I/O
  auto base = builder.addModuleInput("base", Type::index());
  auto offset = builder.addModuleInput("offset", Type::index());
  auto operand = builder.addModuleInput("operand", Type::i32());
  auto result = builder.addModuleOutput("result", Type::i32());
  auto done = builder.addModuleOutput("done", Type::none());

  // addr_gen PE: base + offset -> ld_addr for memory
  builder.connectToModuleInput(base, ag0, 0);
  builder.connectToModuleInput(offset, ag0, 1);
  builder.connectPorts(ag0, 0, mem0, 0); // addr -> memory ld_addr

  // memory lddata -> compute PE + operand -> output
  builder.connectPorts(mem0, 0, comp0, 0); // lddata -> compute input 0
  builder.connectToModuleInput(operand, comp0, 1);
  builder.connectToModuleOutput(comp0, 0, result);
  builder.connectToModuleOutput(mem0, 1, done); // lddone

  builder.exportMLIR("Output/mixed_memory_network.fabric.mlir");
  return 0;
}
