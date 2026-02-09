//===-- pe_constant.cpp - SV test: constant PE module ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("pe_constant");

  auto cpe = builder.newConstantPE("const42")
      .setOutputType(Type::i32())
      .setLatency(0, 0, 0);

  auto c0 = builder.clone(cpe, "c0");

  auto ctrl = builder.addModuleInput("ctrl", Type::none());
  auto out = builder.addModuleOutput("out", Type::i32());

  builder.connectToModuleInput(ctrl, c0, 0);
  builder.connectToModuleOutput(c0, 0, out);

  builder.exportMLIR("Output/pe_constant.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
