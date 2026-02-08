//===-- mixed_pe_switch_mem.cpp - ADG test: PE + switch + memory -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_pe_switch_mem");

  auto pe = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(3, 3)
      .setType(Type::i32());

  auto mem = builder.newMemory("sram")
      .setLoadPorts(1)
      .setStorePorts(0)
      .setShape(MemrefType::static1D(64, Type::i32()));

  auto pe0 = builder.clone(pe, "pe0");
  auto sw0 = builder.clone(sw, "sw0");
  auto mem0 = builder.clone(mem, "mem0");

  // Module I/O
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto sw_in1 = builder.addModuleInput("sw_in1", Type::i32());
  auto sw_in2 = builder.addModuleInput("sw_in2", Type::i32());
  auto ld_addr = builder.addModuleInput("ld_addr", Type::index());
  auto result = builder.addModuleOutput("result", Type::i32());
  auto lddata = builder.addModuleOutput("lddata", Type::i32());
  auto lddone = builder.addModuleOutput("lddone", Type::none());

  // PE: a + b -> switch input 0
  builder.connectToModuleInput(a, pe0, 0);
  builder.connectToModuleInput(b, pe0, 1);
  builder.connectPorts(pe0, 0, sw0, 0);

  // All switch inputs must be connected
  builder.connectToModuleInput(sw_in1, sw0, 1);
  builder.connectToModuleInput(sw_in2, sw0, 2);

  // Switch outputs -> module outputs
  builder.connectToModuleOutput(sw0, 0, result);
  auto sw_out1 = builder.addModuleOutput("sw_out1", Type::i32());
  auto sw_out2 = builder.addModuleOutput("sw_out2", Type::i32());
  builder.connectToModuleOutput(sw0, 1, sw_out1);
  builder.connectToModuleOutput(sw0, 2, sw_out2);

  // Memory: separate load path
  builder.connectToModuleInput(ld_addr, mem0, 0);
  builder.connectToModuleOutput(mem0, 0, lddata);
  builder.connectToModuleOutput(mem0, 1, lddone);

  builder.exportMLIR("Output/mixed_pe_switch_mem.fabric.mlir");
  return 0;
}
