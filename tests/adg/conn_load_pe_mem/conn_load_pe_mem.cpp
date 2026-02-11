//===-- conn_load_pe_mem.cpp - ADG test: LoadPE -> memory load -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_load_pe_mem");

  auto lpe = builder.newLoadPE("ld_native")
      .setDataType(Type::i32())
      .setInterfaceCategory(InterfaceCategory::Native);

  auto mem = builder.newMemory("sram")
      .setLoadPorts(1)
      .setStorePorts(0)
      .setShape(MemrefType::static1D(64, Type::i32()));

  auto ld0 = builder.clone(lpe, "ld0");
  auto mem0 = builder.clone(mem, "mem0");

  // LoadPE inputs: [addr, data_in, ctrl]
  auto addr = builder.addModuleInput("addr", Type::index());
  auto data_in = builder.addModuleInput("data_in", Type::i32());
  auto ctrl = builder.addModuleInput("ctrl", Type::none());
  auto data_out = builder.addModuleOutput("data_out", Type::i32());
  auto lddata = builder.addModuleOutput("lddata", Type::i32());
  auto done = builder.addModuleOutput("done", Type::none());

  builder.connectToModuleInput(addr, ld0, 0);
  builder.connectToModuleInput(data_in, ld0, 1);
  builder.connectToModuleInput(ctrl, ld0, 2);
  // LoadPE outputs: [addr_to_mem, data_to_comp]
  builder.connectToModuleOutput(ld0, 1, data_out);
  // LoadPE addr_to_mem -> memory ld_addr
  builder.connectPorts(ld0, 0, mem0, 0);
  // Memory outputs: [lddata, lddone]
  builder.connectToModuleOutput(mem0, 0, lddata);
  builder.connectToModuleOutput(mem0, 1, done);

  builder.exportMLIR("Output/conn_load_pe_mem.fabric.mlir");
  return 0;
}
