//===-- temporal_pe_reg.cpp - SV test: temporal PE registers -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Tests temporal PE register data flow: write values to register FIFOs, read
// them back, verify FIFO ordering, and test dual-register reads. Uses a single
// arith.addi FU with tagged(i32, i4) interface, NUM_REGISTERS=2,
// NUM_INSTRUCTIONS=4, REG_FIFO_DEPTH=2.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("temporal_pe_reg");

  // Single FU: adder with i32 ports
  auto adder = builder.newPE("fu_add32")
      .addOp("arith.addi")
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .setLatency(1, 1, 1);

  // Temporal PE: interface tagged(i32, i4), 2 registers, 4 instructions
  auto tpe = builder.newTemporalPE("tpe0")
      .setInterface(Type::tagged(Type::i32(), Type::iN(4)))
      .setNumRegisters(2)
      .setNumInstructions(4)
      .setRegFifoDepth(2)
      .addFU(adder);

  auto t0 = builder.clone(tpe, "t0");

  // Module ports: tagged(i32, i4) = 36-bit
  auto in0 = builder.addModuleInput("in0",
      Type::tagged(Type::i32(), Type::iN(4)));
  auto in1 = builder.addModuleInput("in1",
      Type::tagged(Type::i32(), Type::iN(4)));
  auto out = builder.addModuleOutput("out",
      Type::tagged(Type::i32(), Type::iN(4)));

  builder.connectToModuleInput(in0, t0, 0);
  builder.connectToModuleInput(in1, t0, 1);
  builder.connectToModuleOutput(t0, 0, out);

  builder.exportMLIR("Output/temporal_pe_reg.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
