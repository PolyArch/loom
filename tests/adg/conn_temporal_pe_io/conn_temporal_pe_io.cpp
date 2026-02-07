//===-- conn_temporal_pe_io.cpp - ADG test: temporal PE with add/del tag -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_temporal_pe_io");

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

  auto at = builder.newAddTag("tagger")
      .setValueType(Type::i32())
      .setTagType(tagType);

  auto dt = builder.newDelTag("untagger")
      .setInputType(taggedType);

  auto at0 = builder.clone(at, "tag0");
  auto at1 = builder.clone(at, "tag1");
  auto tpe0 = builder.clone(tpe, "tpe0");
  auto dt0 = builder.clone(dt, "untag0");

  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  // add_tag before temporal PE inputs
  builder.connectToModuleInput(a, at0, 0);
  builder.connectToModuleInput(b, at1, 0);
  builder.connectPorts(at0, 0, tpe0, 0);
  builder.connectPorts(at1, 0, tpe0, 1);
  // del_tag after temporal PE output
  builder.connectPorts(tpe0, 0, dt0, 0);
  builder.connectToModuleOutput(dt0, 0, out);

  builder.exportMLIR("Output/conn_temporal_pe_io.fabric.mlir");
  return 0;
}
