//===-- conn_module_io_memref.cpp - ADG test: module memref I/O -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_module_io_memref");

  auto emem = builder.newExtMemory("dram")
      .setLoadPorts(1)
      .setStorePorts(0)
      .setShape(MemrefType::dynamic1D(Type::i32()));

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto em0 = builder.clone(emem, "em0");
  auto add0 = builder.clone(adder, "add0");

  // Static memref module input passed to extmemory
  auto mref = builder.addModuleInput("mem", MemrefType::static1D(256, Type::i32()));
  auto addr = builder.addModuleInput("addr", Type::index());
  auto b = builder.addModuleInput("b", Type::i32());
  auto result = builder.addModuleOutput("result", Type::i32());
  auto done = builder.addModuleOutput("done", Type::none());

  // ExtMemory inputs: [memref, ld_addr]
  builder.connectToModuleInput(mref, em0, 0);
  builder.connectToModuleInput(addr, em0, 1);
  // ExtMemory outputs: [lddata, lddone]
  builder.connectPorts(em0, 0, add0, 0); // lddata -> adder
  builder.connectToModuleInput(b, add0, 1);
  builder.connectToModuleOutput(add0, 0, result);
  builder.connectToModuleOutput(em0, 1, done);

  builder.exportMLIR("Output/conn_module_io_memref.fabric.mlir");
  return 0;
}
