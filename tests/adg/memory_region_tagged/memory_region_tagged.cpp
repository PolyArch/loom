//===-- memory_region_tagged.cpp - ADG test: multi-port mem with regions -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("memory_region_tagged");

  auto mem = builder.newMemory("spad_region")
      .setLoadPorts(4)
      .setStorePorts(2)
      .setQueueDepth(4)
      .setNumRegion(3)
      .setShape(MemrefType::static1D(1024, Type::i32()));

  auto inst = builder.clone(mem, "mem0");

  // Per-port model: each port is an individual untagged port
  // Store inputs
  auto st_data_0 = builder.addModuleInput("st_data_0", Type::i32());
  auto st_addr_0 = builder.addModuleInput("st_addr_0", Type::index());
  auto st_data_1 = builder.addModuleInput("st_data_1", Type::i32());
  auto st_addr_1 = builder.addModuleInput("st_addr_1", Type::index());
  // Load inputs
  auto ld_addr_0 = builder.addModuleInput("ld_addr_0", Type::index());
  auto ld_addr_1 = builder.addModuleInput("ld_addr_1", Type::index());
  auto ld_addr_2 = builder.addModuleInput("ld_addr_2", Type::index());
  auto ld_addr_3 = builder.addModuleInput("ld_addr_3", Type::index());

  // Outputs
  auto ld_data_0 = builder.addModuleOutput("ld_data_0", Type::i32());
  auto ld_data_1 = builder.addModuleOutput("ld_data_1", Type::i32());
  auto ld_data_2 = builder.addModuleOutput("ld_data_2", Type::i32());
  auto ld_data_3 = builder.addModuleOutput("ld_data_3", Type::i32());
  auto ld_done_0 = builder.addModuleOutput("ld_done_0", Type::none());
  auto ld_done_1 = builder.addModuleOutput("ld_done_1", Type::none());
  auto ld_done_2 = builder.addModuleOutput("ld_done_2", Type::none());
  auto ld_done_3 = builder.addModuleOutput("ld_done_3", Type::none());
  auto st_done_0 = builder.addModuleOutput("st_done_0", Type::none());
  auto st_done_1 = builder.addModuleOutput("st_done_1", Type::none());

  // Memory inputs (ld=4, st=2):
  // [st_data_0(0), st_addr_0(1), st_data_1(2), st_addr_1(3),
  //  ld_addr_0(4), ld_addr_1(5), ld_addr_2(6), ld_addr_3(7)]
  builder.connectToModuleInput(st_data_0, inst, 0);
  builder.connectToModuleInput(st_addr_0, inst, 1);
  builder.connectToModuleInput(st_data_1, inst, 2);
  builder.connectToModuleInput(st_addr_1, inst, 3);
  builder.connectToModuleInput(ld_addr_0, inst, 4);
  builder.connectToModuleInput(ld_addr_1, inst, 5);
  builder.connectToModuleInput(ld_addr_2, inst, 6);
  builder.connectToModuleInput(ld_addr_3, inst, 7);

  // Memory outputs:
  // [ld_data_0(0), ld_data_1(1), ld_data_2(2), ld_data_3(3),
  //  ld_done_0(4), ld_done_1(5), ld_done_2(6), ld_done_3(7),
  //  st_done_0(8), st_done_1(9)]
  builder.connectToModuleOutput(inst, 0, ld_data_0);
  builder.connectToModuleOutput(inst, 1, ld_data_1);
  builder.connectToModuleOutput(inst, 2, ld_data_2);
  builder.connectToModuleOutput(inst, 3, ld_data_3);
  builder.connectToModuleOutput(inst, 4, ld_done_0);
  builder.connectToModuleOutput(inst, 5, ld_done_1);
  builder.connectToModuleOutput(inst, 6, ld_done_2);
  builder.connectToModuleOutput(inst, 7, ld_done_3);
  builder.connectToModuleOutput(inst, 8, st_done_0);
  builder.connectToModuleOutput(inst, 9, st_done_1);

  builder.exportMLIR("Output/memory_region_tagged.fabric.mlir");
  return 0;
}
