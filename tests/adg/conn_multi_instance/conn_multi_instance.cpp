//===-- conn_multi_instance.cpp - ADG test: 4 instances of same PE -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_multi_instance");

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  // 4 instances of same adder, tree reduction
  auto a0 = builder.clone(adder, "a0");
  auto a1 = builder.clone(adder, "a1");
  auto a2 = builder.clone(adder, "a2");
  auto a3 = builder.clone(adder, "a3");

  auto x0 = builder.addModuleInput("x0", Type::i32());
  auto x1 = builder.addModuleInput("x1", Type::i32());
  auto x2 = builder.addModuleInput("x2", Type::i32());
  auto x3 = builder.addModuleInput("x3", Type::i32());
  auto x4 = builder.addModuleInput("x4", Type::i32());
  auto x5 = builder.addModuleInput("x5", Type::i32());
  auto x6 = builder.addModuleInput("x6", Type::i32());
  auto x7 = builder.addModuleInput("x7", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(x0, a0, 0);
  builder.connectToModuleInput(x1, a0, 1);
  builder.connectToModuleInput(x2, a1, 0);
  builder.connectToModuleInput(x3, a1, 1);
  builder.connectPorts(a0, 0, a2, 0);
  builder.connectPorts(a1, 0, a2, 1);
  builder.connectToModuleInput(x4, a3, 0);
  builder.connectPorts(a2, 0, a3, 1);
  builder.connectToModuleOutput(a3, 0, out);

  builder.exportMLIR("Output/conn_multi_instance.fabric.mlir");
  return 0;
}
