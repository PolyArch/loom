//===-- const_pe_i32.cpp - ADG test: constant PE i32 -----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("const_pe_i32");

  auto cpe = builder.newConstantPE("const_i32")
      .setOutputType(Type::i32());

  auto inst = builder.clone(cpe, "c0");

  auto ctrl = builder.addModuleInput("ctrl", Type::none());
  auto out = builder.addModuleOutput("val", Type::i32());

  builder.connectToModuleInput(ctrl, inst, 0);
  builder.connectToModuleOutput(inst, 0, out);

  builder.exportMLIR("Output/const_pe_i32.fabric.mlir");
  return 0;
}
