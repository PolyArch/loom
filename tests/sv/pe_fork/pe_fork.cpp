//===-- pe_fork.cpp - SV test: PE with multi-use SSA input ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Tests the eager fork pattern for multi-use SSA values in genMultiOpBodySV.
// Input %a is used by two different operations, triggering fork wire
// generation with sibling-ready gating.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("pe_fork");

  // PE body: %a feeds two independent operations (triggers eager fork)
  //   %r0 = arith.addi %a, %b : i32
  //   %r1 = arith.subi %a, %c : i32
  auto pe = builder.newPE("fork_pe")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32(), Type::i32()})
      .setBodyMLIR(
          "^bb0(%a: i32, %b: i32, %c: i32):\n"
          "  %r0 = arith.addi %a, %b : i32\n"
          "  %r1 = arith.subi %a, %c : i32\n"
          "  fabric.yield %r0, %r1 : i32, i32\n");

  auto inst = builder.clone(pe, "f0");

  auto in0 = builder.addModuleInput("in0", Type::i32());
  auto in1 = builder.addModuleInput("in1", Type::i32());
  auto in2 = builder.addModuleInput("in2", Type::i32());
  auto out0 = builder.addModuleOutput("out0", Type::i32());
  auto out1 = builder.addModuleOutput("out1", Type::i32());

  builder.connectToModuleInput(in0, inst, 0);
  builder.connectToModuleInput(in1, inst, 1);
  builder.connectToModuleInput(in2, inst, 2);
  builder.connectToModuleOutput(inst, 0, out0);
  builder.connectToModuleOutput(inst, 1, out1);

  builder.exportMLIR("Output/pe_fork.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
