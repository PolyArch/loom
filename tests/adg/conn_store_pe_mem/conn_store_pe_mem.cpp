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
  // StorePE outputs: [addr_out(0), done(1)]
  // Connect StorePE addr_out -> memory st_addr (port 0)
  builder.connectPorts(st0, 0, mem0, 0);
  auto st_done = builder.addModuleOutput("st_pe_done", Type::none());
  builder.connectToModuleOutput(st0, 1, st_done); // StorePE done
  // Memory store-only inputs: [st_addr, st_data]
  builder.connectToModuleInput(data, mem0, 1);
  // Memory outputs (store-only, private, ldCount=0): [lddone(0), stdone(1)]
  auto lddone = builder.addModuleOutput("lddone", Type::none());
  builder.connectToModuleOutput(mem0, 0, lddone); // lddone
  builder.connectToModuleOutput(mem0, 1, done);    // stdone

  builder.exportMLIR("Output/conn_store_pe_mem.fabric.mlir");
  return 0;
}
