//===-- const_pe_f32.cpp - ADG test: constant PE f32 -----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("const_pe_f32");

  auto cpe = builder.newConstantPE("const_f32")
      .setOutputType(Type::f32());

  auto inst = builder.clone(cpe, "c0");

  auto ctrl = builder.addModuleInput("ctrl", Type::none());
  auto out = builder.addModuleOutput("val", Type::f32());

  builder.connectToModuleInput(ctrl, inst, 0);
  builder.connectToModuleOutput(inst, 0, out);

  builder.exportMLIR("Output/const_pe_f32.fabric.mlir");
  return 0;
}
