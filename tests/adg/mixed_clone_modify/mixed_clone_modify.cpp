//===-- mixed_clone_modify.cpp - ADG test: 4 clones in diamond pattern -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_clone_modify");

  auto pe = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  // 4 instances of the same PE definition in diamond pattern
  auto top = builder.clone(pe, "top");
  auto left = builder.clone(pe, "left");
  auto right = builder.clone(pe, "right");
  auto bottom = builder.clone(pe, "bottom");

  // Module I/O
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto result = builder.addModuleOutput("result", Type::i32());

  // Diamond: top fans out to left and right, bottom merges
  // top: a + b
  builder.connectToModuleInput(a, top, 0);
  builder.connectToModuleInput(b, top, 1);

  // left: top_out + c
  builder.connectPorts(top, 0, left, 0);
  builder.connectToModuleInput(c, left, 1);

  // right: top_out + d
  builder.connectPorts(top, 0, right, 0);
  builder.connectToModuleInput(d, right, 1);

  // bottom: left_out + right_out
  builder.connectPorts(left, 0, bottom, 0);
  builder.connectPorts(right, 0, bottom, 1);

  builder.connectToModuleOutput(bottom, 0, result);

  builder.exportMLIR("Output/mixed_clone_modify.fabric.mlir");
  return 0;
}
