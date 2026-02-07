//===-- const_pe_tagged.cpp - ADG test: constant PE tagged -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("const_pe_tagged");

  auto taggedType = Type::tagged(Type::i32(), Type::iN(4));

  auto cpe = builder.newConstantPE("const_tagged_i32")
      .setOutputType(taggedType);

  auto inst = builder.clone(cpe, "c0");

  auto taggedNone = Type::tagged(Type::none(), Type::iN(4));
  auto ctrl = builder.addModuleInput("ctrl", taggedNone);
  auto out = builder.addModuleOutput("val", taggedType);

  builder.connectToModuleInput(ctrl, inst, 0);
  builder.connectToModuleOutput(inst, 0, out);

  builder.exportMLIR("Output/const_pe_tagged.fabric.mlir");
  return 0;
}
