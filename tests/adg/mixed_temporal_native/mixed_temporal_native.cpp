//===-- mixed_temporal_native.cpp - ADG test: temporal + native PE -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_temporal_native");

  auto tagType = Type::iN(4);
  auto taggedType = Type::tagged(Type::i32(), tagType);

  // Native PE as FU for the temporal PE
  auto fu = builder.newPE("fu_add")
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto tpe = builder.newTemporalPE("tpe_add")
      .setNumRegisters(0)
      .setNumInstructions(2)
      .setRegFifoDepth(0)
      .setInterface(taggedType)
      .addFU(fu);

  // Native compute PE
  auto nativePE = builder.newPE("native_mul")
      .setLatency(2, 2, 2)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.muli");

  // Instances
  InstanceHandle at0 = builder.newAddTag("tag0")
      .setValueType(Type::i32()).setTagType(tagType);
  InstanceHandle at1 = builder.newAddTag("tag1")
      .setValueType(Type::i32()).setTagType(tagType);
  auto tpe0 = builder.clone(tpe, "tpe0");
  InstanceHandle dt0 = builder.newDelTag("untag0")
      .setInputType(taggedType);
  auto mul0 = builder.clone(nativePE, "mul0");

  // Module I/O
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto tpe_out = builder.addModuleOutput("tpe_result", Type::i32());
  auto mul_out = builder.addModuleOutput("mul_result", Type::i32());

  // Temporal PE path: a,b -> add_tag -> temporal PE -> del_tag -> output
  builder.connectToModuleInput(a, at0, 0);
  builder.connectToModuleInput(b, at1, 0);
  builder.connectPorts(at0, 0, tpe0, 0);
  builder.connectPorts(at1, 0, tpe0, 1);
  builder.connectPorts(tpe0, 0, dt0, 0);
  builder.connectToModuleOutput(dt0, 0, tpe_out);

  // Native PE path: a,c -> mul -> output
  builder.connectToModuleInput(a, mul0, 0);
  builder.connectToModuleInput(c, mul0, 1);
  builder.connectToModuleOutput(mul0, 0, mul_out);

  builder.exportMLIR("Output/mixed_temporal_native.fabric.mlir");
  return 0;
}
