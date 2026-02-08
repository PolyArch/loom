//===-- mixed_deep_pipeline.cpp - ADG test: 10-stage PE pipeline -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_deep_pipeline");

  auto pe = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  // Create 10 PE instances
  auto pe0 = builder.clone(pe, "pe0");
  auto pe1 = builder.clone(pe, "pe1");
  auto pe2 = builder.clone(pe, "pe2");
  auto pe3 = builder.clone(pe, "pe3");
  auto pe4 = builder.clone(pe, "pe4");
  auto pe5 = builder.clone(pe, "pe5");
  auto pe6 = builder.clone(pe, "pe6");
  auto pe7 = builder.clone(pe, "pe7");
  auto pe8 = builder.clone(pe, "pe8");
  auto pe9 = builder.clone(pe, "pe9");

  // Module I/O: one constant operand shared, one value propagating through
  auto init = builder.addModuleInput("init", Type::i32());
  auto step = builder.addModuleInput("step", Type::i32());
  auto result = builder.addModuleOutput("result", Type::i32());

  // Chain: pe0 -> pe1 -> pe2 -> ... -> pe9
  builder.connectToModuleInput(init, pe0, 0);
  builder.connectToModuleInput(step, pe0, 1);
  builder.connectPorts(pe0, 0, pe1, 0);
  builder.connectToModuleInput(step, pe1, 1);
  builder.connectPorts(pe1, 0, pe2, 0);
  builder.connectToModuleInput(step, pe2, 1);
  builder.connectPorts(pe2, 0, pe3, 0);
  builder.connectToModuleInput(step, pe3, 1);
  builder.connectPorts(pe3, 0, pe4, 0);
  builder.connectToModuleInput(step, pe4, 1);
  builder.connectPorts(pe4, 0, pe5, 0);
  builder.connectToModuleInput(step, pe5, 1);
  builder.connectPorts(pe5, 0, pe6, 0);
  builder.connectToModuleInput(step, pe6, 1);
  builder.connectPorts(pe6, 0, pe7, 0);
  builder.connectToModuleInput(step, pe7, 1);
  builder.connectPorts(pe7, 0, pe8, 0);
  builder.connectToModuleInput(step, pe8, 1);
  builder.connectPorts(pe8, 0, pe9, 0);
  builder.connectToModuleInput(step, pe9, 1);

  builder.connectToModuleOutput(pe9, 0, result);

  builder.exportMLIR("Output/mixed_deep_pipeline.fabric.mlir");
  return 0;
}
