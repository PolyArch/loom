//===-- memory_multi_port.cpp - ADG test: multi-port tagged mem -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("memory_multi_port");

  auto tagType = Type::iN(1);
  auto taggedIndex = Type::tagged(Type::index(), tagType);
  auto taggedData = Type::tagged(Type::i32(), tagType);
  auto taggedNone = Type::tagged(Type::none(), tagType);

  auto mem = builder.newMemory("spad_tagged")
      .setLoadPorts(2)
      .setStorePorts(0)
      .setShape(MemrefType::static1D(64, Type::i32()));

  auto inst = builder.clone(mem, "mem0");

  // Singular tagged load address input (TAG_WIDTH=1 for ldCount=2)
  auto ld_addr = builder.addModuleInput("ld_addr", taggedIndex);
  // Singular tagged outputs: ld_data + ld_done
  auto ld_data = builder.addModuleOutput("ld_data", taggedData);
  auto ld_done = builder.addModuleOutput("ld_done", taggedNone);

  // Memory (ldCount=2, stCount=0): inputs=[ld_addr(0)], outputs=[ld_data(0), ld_done(1)]
  builder.connectToModuleInput(ld_addr, inst, 0);
  builder.connectToModuleOutput(inst, 0, ld_data);
  builder.connectToModuleOutput(inst, 1, ld_done);

  builder.exportMLIR("Output/memory_multi_port.fabric.mlir");
  return 0;
}
