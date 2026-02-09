//===-- del_tag.cpp - SV test: del_tag module ------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("del_tag");

  InstanceHandle d0 = builder.newDelTag("d0")
      .setInputType(Type::tagged(Type::i32(), Type::iN(4)));

  auto in = builder.addModuleInput("in", Type::tagged(Type::i32(), Type::iN(4)));
  auto out = builder.addModuleOutput("out", Type::i32());

  builder.connectToModuleInput(in, d0, 0);
  builder.connectToModuleOutput(d0, 0, out);

  builder.exportMLIR("Output/del_tag.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
