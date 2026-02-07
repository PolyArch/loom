//===-- conn_memory_load.cpp - ADG test: addr PE -> mem load ---*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_memory_load");

  auto mem = builder.newMemory("sram")
      .setLoadPorts(1)
      .setStorePorts(0)
      .setShape(MemrefType::static1D(64, Type::i32()));

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto mem0 = builder.clone(mem, "mem0");
  auto add0 = builder.clone(adder, "add0");

  auto addr = builder.addModuleInput("addr", Type::index());
  auto b = builder.addModuleInput("b", Type::i32());
  auto data_out = builder.addModuleOutput("data_out", Type::i32());
  auto done = builder.addModuleOutput("done", Type::none());

  // Private memory (default): no memref output.
  // Outputs: [lddata, lddone]
  builder.connectToModuleInput(addr, mem0, 0);
  builder.connectPorts(mem0, 0, add0, 0); // lddata -> adder
  builder.connectToModuleInput(b, add0, 1);
  builder.connectToModuleOutput(add0, 0, data_out);
  builder.connectToModuleOutput(mem0, 1, done); // lddone

  builder.exportMLIR("Output/conn_memory_load.fabric.mlir");
  return 0;
}
