//===-- temporal_pe_multi_fu.cpp - ADG test: temporal PE 2 FUs -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("temporal_pe_multi_fu");

  auto taggedType = Type::tagged(Type::i32(), Type::iN(4));

  auto fuAdd = builder.newPE("fu_add")
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto fuMul = builder.newPE("fu_mul")
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.muli");

  auto tpe = builder.newTemporalPE("tpe_2fu")
      .setNumRegisters(0)
      .setNumInstructions(4)
      .setRegFifoDepth(0)
      .setInterface(taggedType)
      .addFU(fuAdd)
      .addFU(fuMul);

  auto inst = builder.clone(tpe, "tpe0");

  auto in0 = builder.addModuleInput("a", taggedType);
  auto in1 = builder.addModuleInput("b", taggedType);
  auto out0 = builder.addModuleOutput("r", taggedType);

  builder.connectToModuleInput(in0, inst, 0);
  builder.connectToModuleInput(in1, inst, 1);
  builder.connectToModuleOutput(inst, 0, out0);

  builder.exportMLIR("Output/temporal_pe_multi_fu.fabric.mlir");
  return 0;
}
