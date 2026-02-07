//===-- mixed_f16_network.cpp - ADG test: 3 PEs using f16 type -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_f16_network");

  auto pe = builder.newPE("addf_f16")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::f16(), Type::f16()})
      .setOutputPorts({Type::f16()})
      .addOp("arith.addf");

  auto pe0 = builder.clone(pe, "pe0");
  auto pe1 = builder.clone(pe, "pe1");
  auto pe2 = builder.clone(pe, "pe2");

  auto a = builder.addModuleInput("a", Type::f16());
  auto b = builder.addModuleInput("b", Type::f16());
  auto c = builder.addModuleInput("c", Type::f16());
  auto d = builder.addModuleInput("d", Type::f16());
  auto result = builder.addModuleOutput("result", Type::f16());

  builder.connectToModuleInput(a, pe0, 0);
  builder.connectToModuleInput(b, pe0, 1);
  builder.connectToModuleInput(c, pe1, 0);
  builder.connectToModuleInput(d, pe1, 1);
  builder.connectPorts(pe0, 0, pe2, 0);
  builder.connectPorts(pe1, 0, pe2, 1);
  builder.connectToModuleOutput(pe2, 0, result);

  builder.exportMLIR("Output/mixed_f16_network.fabric.mlir");
  return 0;
}
