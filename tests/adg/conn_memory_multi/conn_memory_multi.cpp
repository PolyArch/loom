//===-- conn_memory_multi.cpp - ADG test: multi-port tagged memory -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_memory_multi");

  // 2 load + 2 store -> maxCount=2, tag width = ceil(log2(2)) = 1 bit
  auto tagType = Type::iN(1);
  auto taggedIndex = Type::tagged(Type::index(), tagType);
  auto taggedData = Type::tagged(Type::i32(), tagType);
  auto taggedNone = Type::tagged(Type::none(), tagType);

  // 2 load + 2 store multi-port memory (requires tagged ports)
  auto mem = builder.newMemory("spad_multi")
      .setLoadPorts(2)
      .setStorePorts(2)
      .setQueueDepth(4)
      .setShape(MemrefType::static1D(64, Type::i32()));

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInterfaceCategory(InterfaceCategory::Tagged)
      .setInputPorts({taggedData, taggedData})
      .setOutputPorts({taggedData})
      .addOp("arith.addi");

  auto mem0 = builder.clone(mem, "mem0");
  auto add0 = builder.clone(adder, "add0");
  auto add1 = builder.clone(adder, "add1");

  // Module inputs for load addresses
  auto ld_addr0 = builder.addModuleInput("ld_addr0", taggedIndex);
  auto ld_addr1 = builder.addModuleInput("ld_addr1", taggedIndex);
  // Module inputs for store addresses
  auto st_addr0 = builder.addModuleInput("st_addr0", taggedIndex);
  auto st_addr1 = builder.addModuleInput("st_addr1", taggedIndex);
  // PE outputs feed store data ports
  auto a = builder.addModuleInput("a", taggedData);
  auto b = builder.addModuleInput("b", taggedData);
  auto c = builder.addModuleInput("c", taggedData);
  auto d = builder.addModuleInput("d", taggedData);
  // Module outputs for load data
  auto lddata0 = builder.addModuleOutput("lddata0", taggedData);
  auto lddata1 = builder.addModuleOutput("lddata1", taggedData);
  auto done = builder.addModuleOutput("done", taggedNone);

  // PE add0: a + b -> store data 0
  builder.connectToModuleInput(a, add0, 0);
  builder.connectToModuleInput(b, add0, 1);
  // PE add1: c + d -> store data 1
  builder.connectToModuleInput(c, add1, 0);
  builder.connectToModuleInput(d, add1, 1);

  // Memory inputs: [ld_addr0, ld_addr1, st_addr0, st_addr1, st_data0, st_data1]
  builder.connectToModuleInput(ld_addr0, mem0, 0);
  builder.connectToModuleInput(ld_addr1, mem0, 1);
  builder.connectToModuleInput(st_addr0, mem0, 2);
  builder.connectToModuleInput(st_addr1, mem0, 3);
  builder.connectPorts(add0, 0, mem0, 4); // st_data0
  builder.connectPorts(add1, 0, mem0, 5); // st_data1

  // Memory outputs: [lddata0, lddata1, lddone, stdone]
  builder.connectToModuleOutput(mem0, 0, lddata0);
  builder.connectToModuleOutput(mem0, 1, lddata1);
  builder.connectToModuleOutput(mem0, 2, done);

  builder.exportMLIR("Output/conn_memory_multi.fabric.mlir");
  return 0;
}
