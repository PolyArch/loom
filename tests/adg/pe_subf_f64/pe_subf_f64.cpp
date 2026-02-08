//===-- pe_subf_f64.cpp - ADG test: arith.subf f64 PE ------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("pe_subf_f64");

  auto pe = builder.newPE("subtractor")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::f64(), Type::f64()})
      .setOutputPorts({Type::f64()})
      .addOp("arith.subf");

  auto inst = builder.clone(pe, "sub_0");

  auto in0 = builder.addModuleInput("a", Type::f64());
  auto in1 = builder.addModuleInput("b", Type::f64());
  auto out = builder.addModuleOutput("result", Type::f64());

  builder.connectToModuleInput(in0, inst, 0);
  builder.connectToModuleInput(in1, inst, 1);
  builder.connectToModuleOutput(inst, 0, out);

  builder.exportMLIR("Output/pe_subf_f64.fabric.mlir");
  return 0;
}
