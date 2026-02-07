//===-- mixed_i64_network.cpp - ADG test: 3 PEs using i64 type -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_i64_network");

  auto pe = builder.newPE("add_i64")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i64(), Type::i64()})
      .setOutputPorts({Type::i64()})
      .addOp("arith.addi");

  auto pe0 = builder.clone(pe, "pe0");
  auto pe1 = builder.clone(pe, "pe1");
  auto pe2 = builder.clone(pe, "pe2");

  auto a = builder.addModuleInput("a", Type::i64());
  auto b = builder.addModuleInput("b", Type::i64());
  auto c = builder.addModuleInput("c", Type::i64());
  auto d = builder.addModuleInput("d", Type::i64());
  auto result = builder.addModuleOutput("result", Type::i64());

  builder.connectToModuleInput(a, pe0, 0);
  builder.connectToModuleInput(b, pe0, 1);
  builder.connectToModuleInput(c, pe1, 0);
  builder.connectToModuleInput(d, pe1, 1);
  builder.connectPorts(pe0, 0, pe2, 0);
  builder.connectPorts(pe1, 0, pe2, 1);
  builder.connectToModuleOutput(pe2, 0, result);

  builder.exportMLIR("Output/mixed_i64_network.fabric.mlir");
  return 0;
}
