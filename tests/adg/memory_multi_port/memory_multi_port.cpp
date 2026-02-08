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

  // 2 load address inputs (tagged)
  auto ld0 = builder.addModuleInput("ld0", taggedIndex);
  auto ld1 = builder.addModuleInput("ld1", taggedIndex);
  // 2 load data outputs (tagged) + 1 done (tagged)
  auto d0 = builder.addModuleOutput("d0", taggedData);
  auto d1 = builder.addModuleOutput("d1", taggedData);
  auto done = builder.addModuleOutput("done", taggedNone);

  builder.connectToModuleInput(ld0, inst, 0);
  builder.connectToModuleInput(ld1, inst, 1);
  builder.connectToModuleOutput(inst, 0, d0);
  builder.connectToModuleOutput(inst, 1, d1);
  builder.connectToModuleOutput(inst, 2, done);

  builder.exportMLIR("Output/memory_multi_port.fabric.mlir");
  return 0;
}
