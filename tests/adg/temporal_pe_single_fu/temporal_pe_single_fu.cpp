//===-- temporal_pe_single_fu.cpp - ADG test: temporal PE 1 FU -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("temporal_pe_single_fu");

  auto taggedType = Type::tagged(Type::i32(), Type::iN(4));

  auto fu = builder.newPE("fu_add")
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto tpe = builder.newTemporalPE("tpe_1fu")
      .setNumRegisters(0)
      .setNumInstructions(2)
      .setRegFifoDepth(0)
      .setInterface(taggedType)
      .addFU(fu);

  auto inst = builder.clone(tpe, "tpe0");

  auto in0 = builder.addModuleInput("a", taggedType);
  auto in1 = builder.addModuleInput("b", taggedType);
  auto out0 = builder.addModuleOutput("r", taggedType);

  builder.connectToModuleInput(in0, inst, 0);
  builder.connectToModuleInput(in1, inst, 1);
  builder.connectToModuleOutput(inst, 0, out0);

  builder.exportMLIR("Output/temporal_pe_single_fu.fabric.mlir");
  return 0;
}
