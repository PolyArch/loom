//===-- pe_dataflow_carry.cpp - SV test: dataflow carry PE ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("pe_dataflow_carry");

  auto pe = builder.newPE("carry_pe")
      .setLatency(0, 0, 0)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i1(), Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("dataflow.carry");

  auto p0 = builder.clone(pe, "p0");

  auto d = builder.addModuleInput("d", Type::i1());
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto o = builder.addModuleOutput("o", Type::i32());

  builder.connectToModuleInput(d, p0, 0);
  builder.connectToModuleInput(a, p0, 1);
  builder.connectToModuleInput(b, p0, 2);
  builder.connectToModuleOutput(p0, 0, o);

  builder.exportMLIR("Output/pe_dataflow_carry.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
