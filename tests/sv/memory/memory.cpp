//===-- memory.cpp - SV test: memory module --------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("memory");

  // Load PE -> memory load port
  auto lpe = builder.newLoadPE("ld_native")
      .setDataType(Type::i32())
      .setInterfaceCategory(InterfaceCategory::Native);

  // Store PE -> memory store port
  auto spe = builder.newStorePE("st_native")
      .setDataType(Type::i32())
      .setInterfaceCategory(InterfaceCategory::Native);

  // Single-port private memory
  auto mem = builder.newMemory("spm")
      .setLoadPorts(1)
      .setStorePorts(1)
      .setQueueDepth(4)
      .setPrivate(true)
      .setShape(MemrefType::static1D(64, Type::i32()));

  auto ld0 = builder.clone(lpe, "ld0");
  auto st0 = builder.clone(spe, "st0");
  auto m0 = builder.clone(mem, "m0");

  // LoadPE inputs: [addr(0), data_in(1), ctrl(2)]
  auto ld_addr = builder.addModuleInput("ld_addr", Type::index());
  auto ld_data_in = builder.addModuleInput("ld_data_in", Type::i32());
  auto ld_ctrl = builder.addModuleInput("ld_ctrl", Type::none());
  builder.connectToModuleInput(ld_addr, ld0, 0);
  builder.connectToModuleInput(ld_data_in, ld0, 1);
  builder.connectToModuleInput(ld_ctrl, ld0, 2);

  // LoadPE outputs: [addr_to_mem(0), data_to_comp(1)]
  auto ld_data_out = builder.addModuleOutput("ld_data_out", Type::i32());
  builder.connectToModuleOutput(ld0, 1, ld_data_out);
  // LoadPE addr_to_mem -> memory ld_addr (port 0)
  builder.connectPorts(ld0, 0, m0, 0);

  // StorePE inputs: [addr(0), data(1), ctrl(2)]
  auto st_addr = builder.addModuleInput("st_addr", Type::index());
  auto st_data = builder.addModuleInput("st_data", Type::i32());
  auto st_ctrl = builder.addModuleInput("st_ctrl", Type::none());
  builder.connectToModuleInput(st_addr, st0, 0);
  builder.connectToModuleInput(st_data, st0, 1);
  builder.connectToModuleInput(st_ctrl, st0, 2);

  // StorePE outputs: [addr_to_mem(0), data_to_mem(1)]
  // StorePE -> memory st_addr(1) and st_data(2)
  builder.connectPorts(st0, 0, m0, 1);
  builder.connectPorts(st0, 1, m0, 2);

  // Memory outputs (private, 1 ld, 1 st): [ld_data(0), ld_done(1), st_done(2)]
  auto lddata = builder.addModuleOutput("lddata", Type::i32());
  auto lddone = builder.addModuleOutput("lddone", Type::none());
  auto stdone = builder.addModuleOutput("stdone", Type::none());
  builder.connectToModuleOutput(m0, 0, lddata);
  builder.connectToModuleOutput(m0, 1, lddone);
  builder.connectToModuleOutput(m0, 2, stdone);

  builder.exportMLIR("Output/memory.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
