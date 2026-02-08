//===-- fifo.cpp - SV test: single fifo module ----------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("fifo");

  auto fifo = builder.newFifo("buf")
      .setDepth(1)
      .setType(Type::i32());

  auto f0 = builder.clone(fifo, "f0");

  auto in = builder.addModuleInput("in", Type::i32());
  auto out = builder.addModuleOutput("out", Type::i32());

  builder.connectToModuleInput(in, f0, 0);
  builder.connectToModuleOutput(f0, 0, out);

  builder.exportMLIR("Output/fifo.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
