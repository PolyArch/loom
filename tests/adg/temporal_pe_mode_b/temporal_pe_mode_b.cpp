//===-- temporal_pe_mode_b.cpp - ADG test: temporal PE mode B ---*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("temporal_pe_mode_b");

  auto taggedType = Type::tagged(Type::i32(), Type::iN(4));

  auto fu = builder.newPE("fu_add")
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto tpe = builder.newTemporalPE("tpe_mode_b")
      .setNumRegisters(2)
      .setNumInstructions(4)
      .setRegFifoDepth(2)
      .setInterface(taggedType)
      .addFU(fu)
      .enableShareOperandBuffer(4);

  auto inst = builder.clone(tpe, "tpe0");

  auto in0 = builder.addModuleInput("a", taggedType);
  auto in1 = builder.addModuleInput("b", taggedType);
  auto out0 = builder.addModuleOutput("r", taggedType);

  builder.connectToModuleInput(in0, inst, 0);
  builder.connectToModuleInput(in1, inst, 1);
  builder.connectToModuleOutput(inst, 0, out0);

  builder.exportMLIR("Output/temporal_pe_mode_b.fabric.mlir");
  return 0;
}
