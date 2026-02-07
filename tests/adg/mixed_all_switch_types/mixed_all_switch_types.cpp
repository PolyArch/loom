//===-- mixed_all_switch_types.cpp - ADG test: native + temporal switch -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_all_switch_types");

  auto tagType = Type::iN(4);
  auto taggedType = Type::tagged(Type::i32(), tagType);

  // Native switch
  auto nsw = builder.newSwitch("native_xbar")
      .setPortCount(3, 3)
      .setType(Type::i32());

  // Temporal switch
  auto tsw = builder.newTemporalSwitch("temporal_xbar")
      .setNumRouteTable(2)
      .setPortCount(3, 3)
      .setInterface(taggedType);

  auto nsw0 = builder.clone(nsw, "nsw0");
  auto tsw0 = builder.clone(tsw, "tsw0");

  // Module I/O for native switch
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto nc = builder.addModuleInput("nc", Type::i32());
  auto nr0 = builder.addModuleOutput("nr0", Type::i32());
  auto nr1 = builder.addModuleOutput("nr1", Type::i32());

  // Module I/O for temporal switch
  auto ta = builder.addModuleInput("ta", taggedType);
  auto tb = builder.addModuleInput("tb", taggedType);
  auto tc = builder.addModuleInput("tc", taggedType);
  auto tr0 = builder.addModuleOutput("tr0", taggedType);
  auto tr1 = builder.addModuleOutput("tr1", taggedType);

  // Native switch connections (all 3 inputs must be connected)
  builder.connectToModuleInput(a, nsw0, 0);
  builder.connectToModuleInput(b, nsw0, 1);
  builder.connectToModuleInput(nc, nsw0, 2);
  builder.connectToModuleOutput(nsw0, 0, nr0);
  builder.connectToModuleOutput(nsw0, 1, nr1);
  auto nr2 = builder.addModuleOutput("nr2", Type::i32());
  builder.connectToModuleOutput(nsw0, 2, nr2);

  // Temporal switch connections (all 3 inputs must be connected)
  builder.connectToModuleInput(ta, tsw0, 0);
  builder.connectToModuleInput(tb, tsw0, 1);
  builder.connectToModuleInput(tc, tsw0, 2);
  builder.connectToModuleOutput(tsw0, 0, tr0);
  builder.connectToModuleOutput(tsw0, 1, tr1);
  auto tr2 = builder.addModuleOutput("tr2", taggedType);
  builder.connectToModuleOutput(tsw0, 2, tr2);

  builder.exportMLIR("Output/mixed_all_switch_types.fabric.mlir");
  return 0;
}
