//===-- mixed_i8_network.cpp - ADG test: 3 PEs using i8 type -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_i8_network");

  auto pe = builder.newPE("add_i8")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i8(), Type::i8()})
      .setOutputPorts({Type::i8()})
      .addOp("arith.addi");

  auto pe0 = builder.clone(pe, "pe0");
  auto pe1 = builder.clone(pe, "pe1");
  auto pe2 = builder.clone(pe, "pe2");

  auto a = builder.addModuleInput("a", Type::i8());
  auto b = builder.addModuleInput("b", Type::i8());
  auto c = builder.addModuleInput("c", Type::i8());
  auto d = builder.addModuleInput("d", Type::i8());
  auto result = builder.addModuleOutput("result", Type::i8());

  // pe0: a + b
  builder.connectToModuleInput(a, pe0, 0);
  builder.connectToModuleInput(b, pe0, 1);
  // pe1: c + d
  builder.connectToModuleInput(c, pe1, 0);
  builder.connectToModuleInput(d, pe1, 1);
  // pe2: pe0 + pe1
  builder.connectPorts(pe0, 0, pe2, 0);
  builder.connectPorts(pe1, 0, pe2, 1);
  builder.connectToModuleOutput(pe2, 0, result);

  builder.exportMLIR("Output/mixed_i8_network.fabric.mlir");
  return 0;
}
