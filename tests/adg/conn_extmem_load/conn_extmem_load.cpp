//===-- conn_extmem_load.cpp - ADG test: extmem load pipeline -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_extmem_load");

  auto emem = builder.newExtMemory("dram_ld")
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

  // ExtMemory first input is memref
  auto mref = builder.addModuleInput("mem", MemrefType::dynamic1D(Type::i32()));
  auto addr = builder.addModuleInput("addr", Type::index());
  auto b = builder.addModuleInput("b", Type::i32());
  auto data_out = builder.addModuleOutput("data_out", Type::i32());
  auto done = builder.addModuleOutput("done", Type::none());

  // ExtMemory inputs: [memref, ld_addr]
  builder.connectToModuleInput(mref, em0, 0);
  builder.connectToModuleInput(addr, em0, 1);
  // ExtMemory outputs: [lddata, lddone]
  builder.connectPorts(em0, 0, add0, 0); // lddata -> adder
  builder.connectToModuleInput(b, add0, 1);
  builder.connectToModuleOutput(add0, 0, data_out);
  builder.connectToModuleOutput(em0, 1, done); // lddone

  builder.exportMLIR("Output/conn_extmem_load.fabric.mlir");
  return 0;
}
