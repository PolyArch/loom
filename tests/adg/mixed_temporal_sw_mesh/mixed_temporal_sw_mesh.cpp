//===-- mixed_temporal_sw_mesh.cpp - ADG test: temporal switch + PE chain -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_temporal_sw_mesh");

  auto tagType = Type::iN(4);
  auto taggedType = Type::tagged(Type::i32(), tagType);

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

  auto tsw = builder.newTemporalSwitch("tsw")
      .setNumRouteTable(2)
      .setPortCount(2, 2)
      .setInterface(taggedType);

  // Chain: tsw0 -> tpe0 -> tsw1 -> tpe1
  auto tsw0 = builder.clone(tsw, "tsw0");
  auto tpe0 = builder.clone(tpe, "tpe0");
  auto tsw1 = builder.clone(tsw, "tsw1");
  auto tpe1 = builder.clone(tpe, "tpe1");

  auto in0 = builder.addModuleInput("a", taggedType);
  auto in1 = builder.addModuleInput("b", taggedType);
  auto out0 = builder.addModuleOutput("r0", taggedType);
  auto out1 = builder.addModuleOutput("r1", taggedType);

  // tsw0: 2 inputs from module
  builder.connectToModuleInput(in0, tsw0, 0);
  builder.connectToModuleInput(in1, tsw0, 1);

  // tsw0 outputs -> tpe0 inputs
  builder.connectPorts(tsw0, 0, tpe0, 0);
  builder.connectPorts(tsw0, 1, tpe0, 1);

  // tpe0 output -> tsw1 input 0, module in1 -> tsw1 input 1
  builder.connectPorts(tpe0, 0, tsw1, 0);
  builder.connectToModuleInput(in1, tsw1, 1);

  // tsw1 outputs -> tpe1 inputs
  builder.connectPorts(tsw1, 0, tpe1, 0);
  builder.connectPorts(tsw1, 1, tpe1, 1);

  // tpe1 output -> module outputs
  builder.connectToModuleOutput(tpe1, 0, out0);
  builder.connectToModuleOutput(tsw1, 0, out1);

  builder.exportMLIR("Output/mixed_temporal_sw_mesh.fabric.mlir");
  return 0;
}
