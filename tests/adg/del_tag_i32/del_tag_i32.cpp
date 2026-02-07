//===-- del_tag_i32.cpp - ADG test: remove tag from tagged i32 -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("del_tag_i32");

  InstanceHandle inst = builder.newDelTag("dt0")
      .setInputType(Type::tagged(Type::i32(), Type::iN(4)));

  auto in0 = builder.addModuleInput("tagged",
      Type::tagged(Type::i32(), Type::iN(4)));
  auto out0 = builder.addModuleOutput("val", Type::i32());

  builder.connectToModuleInput(in0, inst, 0);
  builder.connectToModuleOutput(inst, 0, out0);

  builder.exportMLIR("Output/del_tag_i32.fabric.mlir");
  return 0;
}
