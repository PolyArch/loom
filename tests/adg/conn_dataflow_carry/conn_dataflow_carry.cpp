//===-- conn_dataflow_carry.cpp - ADG test: two-adder accumulate -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_dataflow_carry");

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  // Two separate adders: add0(a, b) -> c, add1(c, d) -> result
  auto add0 = builder.clone(adder, "add0");
  auto add1 = builder.clone(adder, "add1");

  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  // add0: a + b -> c
  builder.connectToModuleInput(a, add0, 0);
  builder.connectToModuleInput(b, add0, 1);
  // add1: c + d -> result
  builder.connectPorts(add0, 0, add1, 0);
  builder.connectToModuleInput(d, add1, 1);
  builder.connectToModuleOutput(add1, 0, out);

  builder.exportMLIR("Output/conn_dataflow_carry.fabric.mlir");
  return 0;
}
