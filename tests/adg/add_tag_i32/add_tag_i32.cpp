//===-- add_tag_i32.cpp - ADG test: add i4 tag to i32 ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("add_tag_i32");

  InstanceHandle inst = builder.newAddTag("at0")
      .setValueType(Type::i32())
      .setTagType(Type::iN(4));

  auto in0 = builder.addModuleInput("val", Type::i32());
  auto out0 = builder.addModuleOutput("tagged",
      Type::tagged(Type::i32(), Type::iN(4)));

  builder.connectToModuleInput(in0, inst, 0);
  builder.connectToModuleOutput(inst, 0, out0);

  builder.exportMLIR("Output/add_tag_i32.fabric.mlir");
  return 0;
}
