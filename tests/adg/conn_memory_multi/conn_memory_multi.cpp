//===-- conn_memory_multi.cpp - ADG test: multi-port tagged memory -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

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

  // Module inputs: singular tagged ports for memory
  auto ld_addr = builder.addModuleInput("ld_addr", taggedIndex);
  auto st_addr = builder.addModuleInput("st_addr", taggedIndex);
  // PE inputs
  auto a = builder.addModuleInput("a", taggedData);
  auto b = builder.addModuleInput("b", taggedData);
  // Module outputs
  auto lddata = builder.addModuleOutput("lddata", taggedData);
  auto lddone = builder.addModuleOutput("lddone", taggedNone);
  auto stdone = builder.addModuleOutput("stdone", taggedNone);

  // PE add0: a + b -> store data
  builder.connectToModuleInput(a, add0, 0);
  builder.connectToModuleInput(b, add0, 1);

  // Memory (ldCount=2, stCount=2): inputs=[ld_addr(0), st_addr(1), st_data(2)]
  builder.connectToModuleInput(ld_addr, mem0, 0);
  builder.connectToModuleInput(st_addr, mem0, 1);
  builder.connectPorts(add0, 0, mem0, 2); // st_data

  // Memory outputs: [ld_data(0), ld_done(1), st_done(2)]
  builder.connectToModuleOutput(mem0, 0, lddata);
  builder.connectToModuleOutput(mem0, 1, lddone);
  builder.connectToModuleOutput(mem0, 2, stdone);

  builder.exportMLIR("Output/conn_memory_multi.fabric.mlir");
  return 0;
}
