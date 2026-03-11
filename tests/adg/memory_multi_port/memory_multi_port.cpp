//===-- memory_multi_port.cpp - ADG test: multi-port memory ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("memory_multi_port");

  auto mem = builder.newMemory("spad_multi_ld")
      .setLoadPorts(2)
      .setStorePorts(0)
      .setShape(MemrefType::static1D(64, Type::i32()));

  auto inst = builder.clone(mem, "mem0");

  // Per-port model: each load port is an individual untagged port
  auto ld_addr_0 = builder.addModuleInput("ld_addr_0", Type::index());
  auto ld_addr_1 = builder.addModuleInput("ld_addr_1", Type::index());
  auto ld_data_0 = builder.addModuleOutput("ld_data_0", Type::i32());
  auto ld_data_1 = builder.addModuleOutput("ld_data_1", Type::i32());
  auto ld_done_0 = builder.addModuleOutput("ld_done_0", Type::none());
  auto ld_done_1 = builder.addModuleOutput("ld_done_1", Type::none());

  // Memory inputs (ld=2, st=0): [ld_addr_0(0), ld_addr_1(1)]
  builder.connectToModuleInput(ld_addr_0, inst, 0);    // ld_addr_0
  builder.connectToModuleInput(ld_addr_1, inst, 1);    // ld_addr_1
  // Memory outputs: [ld_data_0(0), ld_data_1(1), ld_done_0(2), ld_done_1(3)]
  builder.connectToModuleOutput(inst, 0, ld_data_0);   // ld_data_0
  builder.connectToModuleOutput(inst, 1, ld_data_1);   // ld_data_1
  builder.connectToModuleOutput(inst, 2, ld_done_0);   // ld_done_0
  builder.connectToModuleOutput(inst, 3, ld_done_1);   // ld_done_1

  builder.exportMLIR("Output/memory_multi_port.fabric.mlir");
  return 0;
}
