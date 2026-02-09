//===-- add_tag.cpp - SV test: add_tag module ------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("add_tag");

  InstanceHandle t0 = builder.newAddTag("t0")
      .setValueType(Type::i32())
      .setTagType(Type::iN(4));

  auto in = builder.addModuleInput("in", Type::i32());
  auto out = builder.addModuleOutput("out", Type::tagged(Type::i32(), Type::iN(4)));

  builder.connectToModuleInput(in, t0, 0);
  builder.connectToModuleOutput(t0, 0, out);

  builder.exportMLIR("Output/add_tag.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
