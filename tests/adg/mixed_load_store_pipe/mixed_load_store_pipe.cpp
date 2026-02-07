//===-- mixed_load_store_pipe.cpp - ADG test: load-compute-store pipeline -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_load_store_pipe");

  auto pe = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto memLd = builder.newMemory("sram_ld")
      .setLoadPorts(1)
      .setStorePorts(0)
      .setShape(MemrefType::static1D(64, Type::i32()));

  auto memSt = builder.newMemory("sram_st")
      .setLoadPorts(0)
      .setStorePorts(1)
      .setQueueDepth(4)
      .setShape(MemrefType::static1D(64, Type::i32()));

  auto mem_ld0 = builder.clone(memLd, "mem_ld0");
  auto add0 = builder.clone(pe, "add0");
  auto mem_st0 = builder.clone(memSt, "mem_st0");

  // Module I/O
  auto ld_addr = builder.addModuleInput("ld_addr", Type::index());
  auto operand = builder.addModuleInput("operand", Type::i32());
  auto st_addr = builder.addModuleInput("st_addr", Type::index());
  auto lddata = builder.addModuleOutput("lddata", Type::i32());
  auto lddone = builder.addModuleOutput("lddone", Type::none());
  auto stdone = builder.addModuleOutput("stdone", Type::none());

  // Load memory: ld_addr -> lddata
  builder.connectToModuleInput(ld_addr, mem_ld0, 0);
  builder.connectToModuleOutput(mem_ld0, 0, lddata);
  builder.connectToModuleOutput(mem_ld0, 1, lddone);

  // Compute: lddata + operand -> result
  builder.connectPorts(mem_ld0, 0, add0, 0);
  builder.connectToModuleInput(operand, add0, 1);

  // Store memory inputs (store-only): [st_addr, st_data]
  builder.connectToModuleInput(st_addr, mem_st0, 0);
  builder.connectPorts(add0, 0, mem_st0, 1); // adder result -> st_data
  // Store memory outputs (store-only, private, ldCount=0): [lddone(0), stdone(1)]
  auto st_lddone = builder.addModuleOutput("st_lddone", Type::none());
  builder.connectToModuleOutput(mem_st0, 0, st_lddone);
  builder.connectToModuleOutput(mem_st0, 1, stdone);

  builder.exportMLIR("Output/mixed_load_store_pipe.fabric.mlir");
  return 0;
}
