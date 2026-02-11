//===-- pe_dataflow_stream.cpp - SV test: dataflow stream PE ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("pe_dataflow_stream");

  auto pe = builder.newPE("stream_pe")
      .setLatency(0, 0, 0)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::index(), Type::index(), Type::index()})
      .setOutputPorts({Type::index(), Type::i1()})
      .addOp("dataflow.stream");

  auto p0 = builder.clone(pe, "p0");

  auto start = builder.addModuleInput("start", Type::index());
  auto step = builder.addModuleInput("step", Type::index());
  auto bound = builder.addModuleInput("bound", Type::index());
  auto idx = builder.addModuleOutput("idx", Type::index());
  auto cont = builder.addModuleOutput("cont", Type::i1());

  builder.connectToModuleInput(start, p0, 0);
  builder.connectToModuleInput(step, p0, 1);
  builder.connectToModuleInput(bound, p0, 2);
  builder.connectToModuleOutput(p0, 0, idx);
  builder.connectToModuleOutput(p0, 1, cont);

  builder.exportMLIR("Output/pe_dataflow_stream.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
