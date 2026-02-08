//===-- mixed_index_type.cpp - ADG test: PEs using index type -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_index_type");

  auto pe = builder.newPE("idx_add")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::index(), Type::index()})
      .setOutputPorts({Type::index()})
      .addOp("arith.addi");

  auto pe0 = builder.clone(pe, "pe0");
  auto pe1 = builder.clone(pe, "pe1");
  auto pe2 = builder.clone(pe, "pe2");

  auto a = builder.addModuleInput("a", Type::index());
  auto b = builder.addModuleInput("b", Type::index());
  auto c = builder.addModuleInput("c", Type::index());
  auto d = builder.addModuleInput("d", Type::index());
  auto result = builder.addModuleOutput("result", Type::index());

  // pe0: a + b, pe1: c + d, pe2: pe0 + pe1
  builder.connectToModuleInput(a, pe0, 0);
  builder.connectToModuleInput(b, pe0, 1);
  builder.connectToModuleInput(c, pe1, 0);
  builder.connectToModuleInput(d, pe1, 1);
  builder.connectPorts(pe0, 0, pe2, 0);
  builder.connectPorts(pe1, 0, pe2, 1);
  builder.connectToModuleOutput(pe2, 0, result);

  builder.exportMLIR("Output/mixed_index_type.fabric.mlir");
  return 0;
}
