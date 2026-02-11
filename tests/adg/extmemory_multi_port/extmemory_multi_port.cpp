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
  // Singular tagged load address input (TAG_WIDTH=1 for ldCount=2)
  auto ld_addr = builder.addModuleInput("ld_addr", taggedIndex);
  auto ld_data = builder.addModuleOutput("ld_data", taggedData);
  auto ld_done = builder.addModuleOutput("ld_done", taggedNone);

  // ExtMemory (ldCount=2, stCount=0): inputs=[memref(0), ld_addr(1)], outputs=[ld_data(0), ld_done(1)]
  builder.connectToModuleInput(mref, inst, 0);
  builder.connectToModuleInput(ld_addr, inst, 1);
  builder.connectToModuleOutput(inst, 0, ld_data);
  builder.connectToModuleOutput(inst, 1, ld_done);

  builder.exportMLIR("Output/extmemory_multi_port.fabric.mlir");
  return 0;
}
