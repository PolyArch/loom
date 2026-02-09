//===-- map_tag.cpp - SV test: map_tag module ------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("map_tag");

  InstanceHandle m0 = builder.newMapTag("m0")
      .setValueType(Type::i32())
      .setInputTagType(Type::iN(4))
      .setOutputTagType(Type::iN(4))
      .setTableSize(4);

  auto in = builder.addModuleInput("in", Type::tagged(Type::i32(), Type::iN(4)));
  auto out = builder.addModuleOutput("out", Type::tagged(Type::i32(), Type::iN(4)));

  builder.connectToModuleInput(in, m0, 0);
  builder.connectToModuleOutput(m0, 0, out);

  builder.exportMLIR("Output/map_tag.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
