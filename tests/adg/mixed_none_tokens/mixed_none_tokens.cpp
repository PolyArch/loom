//===-- mixed_none_tokens.cpp - ADG test: control tokens with none type -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_none_tokens");

  // Use i32 identity PE (adds 0) as a passthrough, since fabric.pe body must
  // contain at least one non-terminator op.
  auto ctrl_pe = builder.newPE("ctrl_pass")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto ctrl0 = builder.clone(ctrl_pe, "ctrl0");
  auto ctrl1 = builder.clone(ctrl_pe, "ctrl1");

  auto in_a = builder.addModuleInput("in_a", Type::i32());
  auto in_b = builder.addModuleInput("in_b", Type::i32());
  auto in_c = builder.addModuleInput("in_c", Type::i32());
  auto out = builder.addModuleOutput("out", Type::i32());

  // Chain: inputs -> ctrl0 -> ctrl1 -> out
  builder.connectToModuleInput(in_a, ctrl0, 0);
  builder.connectToModuleInput(in_b, ctrl0, 1);
  builder.connectPorts(ctrl0, 0, ctrl1, 0);
  builder.connectToModuleInput(in_c, ctrl1, 1);
  builder.connectToModuleOutput(ctrl1, 0, out);

  builder.exportMLIR("Output/mixed_none_tokens.fabric.mlir");
  return 0;
}
