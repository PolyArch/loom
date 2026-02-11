//===-- pe_dataflow_gate.cpp - SV test: dataflow gate PE --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("pe_dataflow_gate");

  auto pe = builder.newPE("gate_pe")
      .setLatency(0, 0, 0)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i1()})
      .setOutputPorts({Type::i32(), Type::i1()})
      .addOp("dataflow.gate");

  auto p0 = builder.clone(pe, "p0");

  auto bv = builder.addModuleInput("bv", Type::i32());
  auto bc = builder.addModuleInput("bc", Type::i1());
  auto av = builder.addModuleOutput("av", Type::i32());
  auto ac = builder.addModuleOutput("ac", Type::i1());

  builder.connectToModuleInput(bv, p0, 0);
  builder.connectToModuleInput(bc, p0, 1);
  builder.connectToModuleOutput(p0, 0, av);
  builder.connectToModuleOutput(p0, 1, ac);

  builder.exportMLIR("Output/pe_dataflow_gate.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
