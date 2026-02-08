//===-- pe_multi_input.cpp - ADG test: 3-input PE ----------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("pe_multi_input");

  auto pe = builder.newPE("select_pe")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i1(), Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .setBodyMLIR(
          "^bb0(%cond: i1, %a: i32, %b: i32):\n"
          "  %r = arith.select %cond, %a, %b : i32\n"
          "  fabric.yield %r : i32\n");

  auto inst = builder.clone(pe, "sel_0");

  auto cond = builder.addModuleInput("cond", Type::i1());
  auto in0 = builder.addModuleInput("a", Type::i32());
  auto in1 = builder.addModuleInput("b", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(cond, inst, 0);
  builder.connectToModuleInput(in0, inst, 1);
  builder.connectToModuleInput(in1, inst, 2);
  builder.connectToModuleOutput(inst, 0, out);

  builder.exportMLIR("Output/pe_multi_input.fabric.mlir");
  return 0;
}
