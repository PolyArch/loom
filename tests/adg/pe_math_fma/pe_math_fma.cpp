//===-- pe_math_fma.cpp - ADG test: math.fma 3-input f32 PE -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("pe_math_fma");

  auto pe = builder.newPE("fma_pe")
      .setLatency(1, 3, 3)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::f32(), Type::f32(), Type::f32()})
      .setOutputPorts({Type::f32()})
      .addOp("math.fma");

  auto inst = builder.clone(pe, "fma_0");

  auto in0 = builder.addModuleInput("a", Type::f32());
  auto in1 = builder.addModuleInput("b", Type::f32());
  auto in2 = builder.addModuleInput("c", Type::f32());
  auto out = builder.addModuleOutput("result", Type::f32());

  builder.connectToModuleInput(in0, inst, 0);
  builder.connectToModuleInput(in1, inst, 1);
  builder.connectToModuleInput(in2, inst, 2);
  builder.connectToModuleOutput(inst, 0, out);

  builder.exportMLIR("Output/pe_math_fma.fabric.mlir");
  return 0;
}
