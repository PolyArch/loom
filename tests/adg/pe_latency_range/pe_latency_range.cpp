//===-- pe_latency_range.cpp - ADG test: non-uniform latency -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("pe_latency_range");

  auto pe = builder.newPE("var_lat_pe")
      .setLatency(1, 2, 3)
      .setInterval(1, 1, 2)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto inst = builder.clone(pe, "pe_0");

  auto in0 = builder.addModuleInput("a", Type::i32());
  auto in1 = builder.addModuleInput("b", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(in0, inst, 0);
  builder.connectToModuleInput(in1, inst, 1);
  builder.connectToModuleOutput(inst, 0, out);

  builder.exportMLIR("Output/pe_latency_range.fabric.mlir");
  return 0;
}
