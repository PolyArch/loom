//===-- conn_memory_multi.cpp - ADG test: multi-port memory ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_memory_multi");

  // 2 load + 2 store multi-port memory (per-port model: individual untagged ports)
  auto mem = builder.newMemory("spad_multi")
      .setLoadPorts(2)
      .setStorePorts(2)
      .setQueueDepth(4)
      .setShape(MemrefType::static1D(64, Type::i32()));

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto mem0 = builder.clone(mem, "mem0");
  auto add0 = builder.clone(adder, "add0");
  auto add1 = builder.clone(adder, "add1");

  // Module inputs
  auto ld_addr_0 = builder.addModuleInput("ld_addr_0", Type::index());
  auto ld_addr_1 = builder.addModuleInput("ld_addr_1", Type::index());
  auto st_addr_0 = builder.addModuleInput("st_addr_0", Type::index());
  auto st_addr_1 = builder.addModuleInput("st_addr_1", Type::index());
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());

  // Module outputs
  auto lddata_0 = builder.addModuleOutput("lddata_0", Type::i32());
  auto lddata_1 = builder.addModuleOutput("lddata_1", Type::i32());
  auto lddone_0 = builder.addModuleOutput("lddone_0", Type::none());
  auto lddone_1 = builder.addModuleOutput("lddone_1", Type::none());
  auto stdone_0 = builder.addModuleOutput("stdone_0", Type::none());
  auto stdone_1 = builder.addModuleOutput("stdone_1", Type::none());

  // PE add0: a + b -> store data for port 0
  builder.connectToModuleInput(a, add0, 0);
  builder.connectToModuleInput(b, add0, 1);
  // PE add1: c + d -> store data for port 1
  builder.connectToModuleInput(c, add1, 0);
  builder.connectToModuleInput(d, add1, 1);

  // Memory inputs (ld=2, st=2):
  // [st_data_0(0), st_addr_0(1), st_data_1(2), st_addr_1(3), ld_addr_0(4), ld_addr_1(5)]
  builder.connectPorts(add0, 0, mem0, 0);              // st_data_0
  builder.connectToModuleInput(st_addr_0, mem0, 1);     // st_addr_0
  builder.connectPorts(add1, 0, mem0, 2);              // st_data_1
  builder.connectToModuleInput(st_addr_1, mem0, 3);     // st_addr_1
  builder.connectToModuleInput(ld_addr_0, mem0, 4);     // ld_addr_0
  builder.connectToModuleInput(ld_addr_1, mem0, 5);     // ld_addr_1

  // Memory outputs (2 ld, 2 st):
  // [ld_data_0(0), ld_data_1(1), ld_done_0(2), ld_done_1(3), st_done_0(4), st_done_1(5)]
  builder.connectToModuleOutput(mem0, 0, lddata_0);    // ld_data_0
  builder.connectToModuleOutput(mem0, 1, lddata_1);    // ld_data_1
  builder.connectToModuleOutput(mem0, 2, lddone_0);    // ld_done_0
  builder.connectToModuleOutput(mem0, 3, lddone_1);    // ld_done_1
  builder.connectToModuleOutput(mem0, 4, stdone_0);    // st_done_0
  builder.connectToModuleOutput(mem0, 5, stdone_1);    // st_done_1

  builder.exportMLIR("Output/conn_memory_multi.fabric.mlir");
  return 0;
}
