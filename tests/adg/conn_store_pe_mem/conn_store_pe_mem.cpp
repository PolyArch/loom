//===-- conn_store_pe_mem.cpp - ADG test: StorePE -> memory store -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

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

  // StorePE inputs: [addr(0), data(1), ctrl(2)]
  auto addr = builder.addModuleInput("addr", Type::index());
  auto data = builder.addModuleInput("data", Type::i32());
  auto ctrl = builder.addModuleInput("ctrl", Type::none());
  auto done = builder.addModuleOutput("done", Type::none());

  builder.connectToModuleInput(addr, st0, 0);
  builder.connectToModuleInput(data, st0, 1);
  builder.connectToModuleInput(ctrl, st0, 2);
  // StorePE outputs: [addr_to_mem(0), data_to_mem(1)]
  // Connect StorePE -> memory st_addr(0) and st_data(1)
  builder.connectPorts(st0, 0, mem0, 0);
  builder.connectPorts(st0, 1, mem0, 1);
  // Memory outputs (ldCount=0, stCount=1): [st_done(0)]
  builder.connectToModuleOutput(mem0, 0, done);

  builder.exportMLIR("Output/conn_store_pe_mem.fabric.mlir");
  return 0;
}
