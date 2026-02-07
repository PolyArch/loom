//===-- load_pe_native.cpp - ADG test: native load PE ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("load_pe_native");

  auto lpe = builder.newLoadPE("ld_native")
      .setDataType(Type::i32())
      .setInterfaceCategory(InterfaceCategory::Native);

  auto inst = builder.clone(lpe, "ld0");

  auto addr = builder.addModuleInput("addr", Type::index());
  auto data_in = builder.addModuleInput("data_in", Type::i32());
  auto ctrl = builder.addModuleInput("ctrl", Type::none());
  auto data_out = builder.addModuleOutput("data_out", Type::i32());
  auto addr_out = builder.addModuleOutput("addr_out", Type::index());

  builder.connectToModuleInput(addr, inst, 0);
  builder.connectToModuleInput(data_in, inst, 1);
  builder.connectToModuleInput(ctrl, inst, 2);
  builder.connectToModuleOutput(inst, 0, data_out);
  builder.connectToModuleOutput(inst, 1, addr_out);

  builder.exportMLIR("Output/load_pe_native.fabric.mlir");
  return 0;
}
