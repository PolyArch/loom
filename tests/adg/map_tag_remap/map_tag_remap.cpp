//===-- map_tag_remap.cpp - ADG test: remap i4 tag to i2 ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("map_tag_remap");

  InstanceHandle inst = builder.newMapTag("mt0")
      .setValueType(Type::i32())
      .setInputTagType(Type::iN(4))
      .setOutputTagType(Type::iN(2))
      .setTableSize(4);

  auto in0 = builder.addModuleInput("in",
      Type::tagged(Type::i32(), Type::iN(4)));
  auto out0 = builder.addModuleOutput("out",
      Type::tagged(Type::i32(), Type::iN(2)));

  builder.connectToModuleInput(in0, inst, 0);
  builder.connectToModuleOutput(inst, 0, out0);

  builder.exportMLIR("Output/map_tag_remap.fabric.mlir");
  return 0;
}
