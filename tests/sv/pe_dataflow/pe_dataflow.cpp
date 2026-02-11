//===-- pe_dataflow.cpp - SV test: dataflow invariant PE --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("pe_dataflow");

  auto pe = builder.newPE("inv_pe")
      .setLatency(0, 0, 0)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i1(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("dataflow.invariant");

  auto p0 = builder.clone(pe, "p0");

  auto d = builder.addModuleInput("d", Type::i1());
  auto a = builder.addModuleInput("a", Type::i32());
  auto o = builder.addModuleOutput("o", Type::i32());

  builder.connectToModuleInput(d, p0, 0);
  builder.connectToModuleInput(a, p0, 1);
  builder.connectToModuleOutput(p0, 0, o);

  builder.exportMLIR("Output/pe_dataflow.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
