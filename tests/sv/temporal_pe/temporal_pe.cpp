//===-- temporal_pe.cpp - SV test: temporal PE module ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("temporal_pe");

  // Create two FU types for the temporal PE body
  auto adder = builder.newPE("fu_add")
      .addOp("arith.addi")
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .setLatency(1, 1, 1);

  auto multiplier = builder.newPE("fu_mul")
      .addOp("arith.muli")
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .setLatency(3, 3, 3);

  auto tpe = builder.newTemporalPE("tpe0")
      .setInterface(Type::tagged(Type::i32(), Type::iN(4)))
      .setNumRegisters(0)
      .setNumInstructions(2)
      .setRegFifoDepth(0)
      .addFU(adder)
      .addFU(multiplier);

  auto t0 = builder.clone(tpe, "t0");

  auto in0 = builder.addModuleInput("in0", Type::tagged(Type::i32(), Type::iN(4)));
  auto in1 = builder.addModuleInput("in1", Type::tagged(Type::i32(), Type::iN(4)));
  auto out = builder.addModuleOutput("out", Type::tagged(Type::i32(), Type::iN(4)));

  builder.connectToModuleInput(in0, t0, 0);
  builder.connectToModuleInput(in1, t0, 1);
  builder.connectToModuleOutput(t0, 0, out);

  builder.exportMLIR("Output/temporal_pe.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
