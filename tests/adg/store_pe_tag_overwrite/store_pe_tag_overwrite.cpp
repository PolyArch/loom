//===-- store_pe_tag_overwrite.cpp - ADG test: tagged store PE -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("store_pe_tag_overwrite");

  auto spe = builder.newStorePE("st_tag_ow")
      .setDataType(Type::i32())
      .setInterfaceCategory(InterfaceCategory::Tagged)
      .setTagWidth(4)
      .setHardwareType(HardwareType::TagOverwrite);

  auto inst = builder.clone(spe, "st0");

  auto tagType = Type::iN(4);
  auto taggedIndex = Type::tagged(Type::index(), tagType);
  auto taggedData = Type::tagged(Type::i32(), tagType);
  auto taggedNone = Type::tagged(Type::none(), tagType);

  auto addr = builder.addModuleInput("addr", taggedIndex);
  auto data = builder.addModuleInput("data", taggedData);
  auto ctrl = builder.addModuleInput("ctrl", taggedNone);
  auto addr_out = builder.addModuleOutput("addr_out", taggedIndex);
  auto done = builder.addModuleOutput("done", taggedNone);

  builder.connectToModuleInput(addr, inst, 0);
  builder.connectToModuleInput(data, inst, 1);
  builder.connectToModuleInput(ctrl, inst, 2);
  builder.connectToModuleOutput(inst, 0, addr_out);
  builder.connectToModuleOutput(inst, 1, done);

  builder.exportMLIR("Output/store_pe_tag_overwrite.fabric.mlir");
  return 0;
}
