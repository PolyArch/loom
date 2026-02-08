//===-- extmemory_multi_port.cpp - ADG test: multi-port extmem -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("extmemory_multi_port");

  auto tagType = Type::iN(1);
  auto taggedIndex = Type::tagged(Type::index(), tagType);
  auto taggedData = Type::tagged(Type::i32(), tagType);
  auto taggedNone = Type::tagged(Type::none(), tagType);

  auto emem = builder.newExtMemory("dram_multi")
      .setLoadPorts(2)
      .setStorePorts(0)
      .setShape(MemrefType::dynamic1D(Type::i32()));

  auto inst = builder.clone(emem, "em0");

  auto mref = builder.addModuleInput("m", MemrefType::dynamic1D(Type::i32()));
  auto ld0 = builder.addModuleInput("ld0", taggedIndex);
  auto ld1 = builder.addModuleInput("ld1", taggedIndex);
  auto d0 = builder.addModuleOutput("d0", taggedData);
  auto d1 = builder.addModuleOutput("d1", taggedData);
  auto done = builder.addModuleOutput("done", taggedNone);

  builder.connectToModuleInput(mref, inst, 0);
  builder.connectToModuleInput(ld0, inst, 1);
  builder.connectToModuleInput(ld1, inst, 2);
  builder.connectToModuleOutput(inst, 0, d0);
  builder.connectToModuleOutput(inst, 1, d1);
  builder.connectToModuleOutput(inst, 2, done);

  builder.exportMLIR("Output/extmemory_multi_port.fabric.mlir");
  return 0;
}
