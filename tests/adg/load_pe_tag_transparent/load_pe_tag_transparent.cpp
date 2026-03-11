//===-- load_pe_tag_transparent.cpp - ADG test: transparent load -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("load_pe_tag_transparent");

  auto lpe = builder.newLoadPE("ld_tag_tp")
      .setDataType(Type::i32())
      .setInterfaceCategory(InterfaceCategory::Tagged)
      .setTagWidth(4)
      .setQueueDepth(8)
      .setHardwareType(HardwareType::TagTransparent);

  auto inst = builder.clone(lpe, "ld0");

  auto tagType = Type::iN(4);
  auto taggedIndex = Type::tagged(Type::index(), tagType);
  auto taggedData = Type::tagged(Type::i32(), tagType);
  auto taggedNone = Type::tagged(Type::none(), tagType);

  auto addr = builder.addModuleInput("addr", taggedIndex);
  auto data_in = builder.addModuleInput("data_in", taggedData);
  auto ctrl = builder.addModuleInput("ctrl", taggedNone);
  auto data_out = builder.addModuleOutput("data_out", taggedData);
  auto addr_out = builder.addModuleOutput("addr_out", taggedIndex);

  builder.connectToModuleInput(addr, inst, 0);
  builder.connectToModuleInput(data_in, inst, 1);
  builder.connectToModuleInput(ctrl, inst, 2);
  builder.connectToModuleOutput(inst, 0, data_out);
  builder.connectToModuleOutput(inst, 1, addr_out);

  builder.exportMLIR("Output/load_pe_tag_transparent.fabric.mlir");
  return 0;
}
