//===-- conn_store_pe_mem.cpp - ADG test: StorePE -> memory store -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_store_pe_mem");

  auto spe = builder.newStorePE("st_native")
      .setDataType(Type::i32())
      .setInterfaceCategory(InterfaceCategory::Native);

  auto mem = builder.newMemory("sram")
      .setLoadPorts(0)
      .setStorePorts(1)
      .setQueueDepth(1)
      .setShape(MemrefType::static1D(64, Type::i32()));

  auto st0 = builder.clone(spe, "st0");
  auto mem0 = builder.clone(mem, "mem0");

  // StorePE inputs: [addr, data, ctrl]
  auto addr = builder.addModuleInput("addr", Type::index());
  auto data = builder.addModuleInput("data", Type::i32());
  auto ctrl = builder.addModuleInput("ctrl", Type::none());
  auto done = builder.addModuleOutput("done", Type::none());

  builder.connectToModuleInput(addr, st0, 0);
  builder.connectToModuleInput(data, st0, 1);
  builder.connectToModuleInput(ctrl, st0, 2);
  // StorePE outputs: [addr_out, done]
  // Connect StorePE addr_out -> memory st_addr (port 0)
  builder.connectPorts(st0, 0, mem0, 0);
  // Connect StorePE done as store data -> memory st_data (port 1)
  // Actually, memory store-only inputs: [st_addr, st_data]
  // We need separate data for memory. Use module input for st_data.
  // StorePE addr_out goes to memory st_addr, module provides st_data separately.
  builder.connectToModuleInput(data, mem0, 1);
  // Memory outputs (store-only): [lddone, stdone] -- lddone still present
  builder.connectToModuleOutput(mem0, 1, done); // stdone

  builder.exportMLIR("Output/conn_store_pe_mem.fabric.mlir");
  return 0;
}
