//===-- memory_tagged.cpp - SV test: tagged load/store memory path -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("memory_tagged");

  auto lpe = builder.newLoadPE("ld_tagged")
      .setDataType(Type::i32())
      .setInterfaceCategory(InterfaceCategory::Tagged)
      .setTagWidth(1)
      .setQueueDepth(4)
      .setHardwareType(HardwareType::TagTransparent);

  auto spe = builder.newStorePE("st_tagged")
      .setDataType(Type::i32())
      .setInterfaceCategory(InterfaceCategory::Tagged)
      .setTagWidth(1)
      .setQueueDepth(4)
      .setHardwareType(HardwareType::TagTransparent);

  auto mem = builder.newMemory("m_tagged")
      .setLoadPorts(2)
      .setStorePorts(2)
      .setQueueDepth(4)
      .setPrivate(true)
      .setShape(MemrefType::static1D(64, Type::i32()));

  auto ld0 = builder.clone(lpe, "ld0");
  auto st0 = builder.clone(spe, "st0");
  auto m0 = builder.clone(mem, "m0");

  auto ld_addr = builder.addModuleInput(
      "ld_addr", Type::tagged(Type::index(), Type::iN(1)));
  auto ld_ctrl = builder.addModuleInput(
      "ld_ctrl", Type::tagged(Type::none(), Type::iN(1)));

  auto st_addr = builder.addModuleInput(
      "st_addr", Type::tagged(Type::index(), Type::iN(1)));
  auto st_data = builder.addModuleInput(
      "st_data", Type::tagged(Type::i32(), Type::iN(1)));
  auto st_ctrl = builder.addModuleInput(
      "st_ctrl", Type::tagged(Type::none(), Type::iN(1)));

  auto ld_out = builder.addModuleOutput(
      "ld_out", Type::tagged(Type::i32(), Type::iN(1)));
  auto lddone = builder.addModuleOutput(
      "lddone", Type::tagged(Type::none(), Type::iN(1)));
  auto stdone = builder.addModuleOutput(
      "stdone", Type::tagged(Type::none(), Type::iN(1)));

  // Load PE input side
  builder.connectToModuleInput(ld_addr, ld0, 0);
  builder.connectToModuleInput(ld_ctrl, ld0, 2);

  // Store PE input side
  builder.connectToModuleInput(st_addr, st0, 0);
  builder.connectToModuleInput(st_data, st0, 1);
  builder.connectToModuleInput(st_ctrl, st0, 2);

  // Memory connections
  builder.connectPorts(ld0, 0, m0, 0); // load address
  builder.connectPorts(st0, 0, m0, 1); // store address
  builder.connectPorts(st0, 1, m0, 2); // store data

  // Memory load data returns through load PE data input
  builder.connectPorts(m0, 0, ld0, 1);

  // Expose load output and done channels
  builder.connectToModuleOutput(ld0, 1, ld_out);
  builder.connectToModuleOutput(m0, 1, lddone);
  builder.connectToModuleOutput(m0, 2, stdone);

  builder.exportMLIR("Output/memory_tagged.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
