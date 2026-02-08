//===-- pe_multi_output.cpp - ADG test: 2-output PE --------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("pe_multi_output");

  auto pe = builder.newPE("dup_pe")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32(), Type::i32()})
      .setBodyMLIR(
          "  %sum = arith.addi %arg0, %arg1 : i32\n"
          "  fabric.yield %sum, %sum : i32, i32\n");

  auto inst = builder.clone(pe, "dup_0");

  auto in0 = builder.addModuleInput("a", Type::i32());
  auto in1 = builder.addModuleInput("b", Type::i32());
  auto out0 = builder.addModuleOutput("r0", Type::i32());
  auto out1 = builder.addModuleOutput("r1", Type::i32());

  builder.connectToModuleInput(in0, inst, 0);
  builder.connectToModuleInput(in1, inst, 1);
  builder.connectToModuleOutput(inst, 0, out0);
  builder.connectToModuleOutput(inst, 1, out1);

  builder.exportMLIR("Output/pe_multi_output.fabric.mlir");
  return 0;
}
