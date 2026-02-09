//===-- pe.cpp - SV test: single-op PE module ------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("pe");

  auto pe = builder.newPE("adder")
      .addOp("arith.addi")
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .setLatency(1, 1, 1);

  auto p0 = builder.clone(pe, "p0");

  auto in0 = builder.addModuleInput("in0", Type::i32());
  auto in1 = builder.addModuleInput("in1", Type::i32());
  auto out = builder.addModuleOutput("out", Type::i32());

  builder.connectToModuleInput(in0, p0, 0);
  builder.connectToModuleInput(in1, p0, 1);
  builder.connectToModuleOutput(p0, 0, out);

  builder.exportMLIR("Output/pe.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
