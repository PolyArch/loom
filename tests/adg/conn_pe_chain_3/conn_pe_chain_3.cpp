//===-- conn_pe_chain_3.cpp - ADG test: chain of 3 PEs --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_pe_chain_3");

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto a0 = builder.clone(adder, "a0");
  auto a1 = builder.clone(adder, "a1");
  auto a2 = builder.clone(adder, "a2");

  auto in0 = builder.addModuleInput("in0", Type::i32());
  auto in1 = builder.addModuleInput("in1", Type::i32());
  auto in2 = builder.addModuleInput("in2", Type::i32());
  auto in3 = builder.addModuleInput("in3", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(in0, a0, 0);
  builder.connectToModuleInput(in1, a0, 1);
  builder.connectPorts(a0, 0, a1, 0);
  builder.connectToModuleInput(in2, a1, 1);
  builder.connectPorts(a1, 0, a2, 0);
  builder.connectToModuleInput(in3, a2, 1);
  builder.connectToModuleOutput(a2, 0, out);

  builder.exportMLIR("Output/conn_pe_chain_3.fabric.mlir");
  return 0;
}
