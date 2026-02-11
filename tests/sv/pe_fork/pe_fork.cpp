//===-- pe_fork.cpp - SV test: PE with multi-use SSA input ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Tests the eager fork pattern for multi-use SSA values in genMultiOpBodySV.
// Input %a is used by two different operations (arith.muli and arith.addi),
// triggering fork wire generation with sibling-ready gating.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("pe_fork");

  // PE body: %a feeds two operations (triggers eager fork)
  //   %sq = arith.muli %a, %a : i32     (%a used twice by same op)
  //   %r  = arith.addi %sq, %b : i32
  auto pe = builder.newPE("fork_pe")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .setBodyMLIR(
          "^bb0(%a: i32, %b: i32):\n"
          "  %sq = arith.muli %a, %a : i32\n"
          "  %r  = arith.addi %sq, %b : i32\n"
          "  fabric.yield %r : i32\n");

  auto inst = builder.clone(pe, "f0");

  auto in0 = builder.addModuleInput("in0", Type::i32());
  auto in1 = builder.addModuleInput("in1", Type::i32());
  auto out = builder.addModuleOutput("out", Type::i32());

  builder.connectToModuleInput(in0, inst, 0);
  builder.connectToModuleInput(in1, inst, 1);
  builder.connectToModuleOutput(inst, 0, out);

  builder.exportMLIR("Output/pe_fork.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
