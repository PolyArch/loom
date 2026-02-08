//===-- pe_mulf_f32.cpp - ADG test: arith.mulf f32 PE ------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("pe_mulf_f32");

  auto pe = builder.newPE("multiplier")
      .setLatency(1, 2, 3)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::f32(), Type::f32()})
      .setOutputPorts({Type::f32()})
      .addOp("arith.mulf");

  auto inst = builder.clone(pe, "mul_0");

  auto in0 = builder.addModuleInput("a", Type::f32());
  auto in1 = builder.addModuleInput("b", Type::f32());
  auto out = builder.addModuleOutput("result", Type::f32());

  builder.connectToModuleInput(in0, inst, 0);
  builder.connectToModuleInput(in1, inst, 1);
  builder.connectToModuleOutput(inst, 0, out);

  builder.exportMLIR("Output/pe_mulf_f32.fabric.mlir");
  return 0;
}
