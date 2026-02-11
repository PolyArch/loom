//===-- load_pe_tag_overwrite.cpp - ADG test: tagged load PE ---*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("load_pe_tag_overwrite");

  auto lpe = builder.newLoadPE("ld_tag_ow")
      .setDataType(Type::i32())
      .setInterfaceCategory(InterfaceCategory::Tagged)
      .setTagWidth(4)
      .setHardwareType(HardwareType::TagOverwrite);

  auto inst = builder.clone(lpe, "ld0");

  auto tagType = Type::iN(4);
  auto taggedIndex = Type::tagged(Type::index(), tagType);
  auto taggedData = Type::tagged(Type::i32(), tagType);
  auto taggedNone = Type::tagged(Type::none(), tagType);

  // TagOverwrite: ctrl port is plain none (not tagged)
  auto addr = builder.addModuleInput("addr", taggedIndex);
  auto data_in = builder.addModuleInput("data_in", taggedData);
  auto ctrl = builder.addModuleInput("ctrl", Type::none());
  auto data_out = builder.addModuleOutput("data_out", taggedData);
  auto addr_out = builder.addModuleOutput("addr_out", taggedIndex);

  builder.connectToModuleInput(addr, inst, 0);
  builder.connectToModuleInput(data_in, inst, 1);
  builder.connectToModuleInput(ctrl, inst, 2);
  builder.connectToModuleOutput(inst, 0, addr_out);
  builder.connectToModuleOutput(inst, 1, data_out);

  builder.exportMLIR("Output/load_pe_tag_overwrite.fabric.mlir");
  return 0;
}
