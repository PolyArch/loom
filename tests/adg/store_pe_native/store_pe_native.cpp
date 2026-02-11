//===-- store_pe_native.cpp - ADG test: native store PE -------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("store_pe_native");

  auto spe = builder.newStorePE("st_native")
      .setDataType(Type::i32())
      .setInterfaceCategory(InterfaceCategory::Native);

  auto inst = builder.clone(spe, "st0");

  auto addr = builder.addModuleInput("addr", Type::index());
  auto data = builder.addModuleInput("data", Type::i32());
  auto ctrl = builder.addModuleInput("ctrl", Type::none());
  auto addr_out = builder.addModuleOutput("addr_out", Type::index());
  auto data_out = builder.addModuleOutput("data_out", Type::i32());

  builder.connectToModuleInput(addr, inst, 0);
  builder.connectToModuleInput(data, inst, 1);
  builder.connectToModuleInput(ctrl, inst, 2);
  builder.connectToModuleOutput(inst, 0, addr_out);
  builder.connectToModuleOutput(inst, 1, data_out);

  builder.exportMLIR("Output/store_pe_native.fabric.mlir");
  return 0;
}
