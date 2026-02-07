//===-- pe_body_mlir.cpp - ADG test: setBodyMLIR with max() ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("pe_body_mlir");

  auto pe = builder.newPE("max_pe")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .setBodyMLIR(
          "^bb0(%a: i32, %b: i32):\n"
          "  %cmp = arith.cmpi sgt, %a, %b : i32\n"
          "  %max = arith.select %cmp, %a, %b : i32\n"
          "  fabric.yield %max : i32\n");

  auto inst = builder.clone(pe, "max_0");

  auto in0 = builder.addModuleInput("x", Type::i32());
  auto in1 = builder.addModuleInput("y", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(in0, inst, 0);
  builder.connectToModuleInput(in1, inst, 1);
  builder.connectToModuleOutput(inst, 0, out);

  builder.exportMLIR("Output/pe_body_mlir.fabric.mlir");
  return 0;
}
