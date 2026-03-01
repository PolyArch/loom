//===-- error_tagged_payload_width_mismatch.cpp - ADG test -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verifies that validateADG detects non-temporal tagged connections where
// the payload widths differ (e.g. tagged<i32, i4> vs tagged<i16, i4>).
// The tag types match but the value-type data widths do not.
//
// Uses native types (i32, i16) as ADG-level value types because the ADG
// builder converts them to bits<N> for MLIR emission. The widthCompatible()
// check in ADGBuilderValidation compares getTypeDataWidth() values, which
// works identically for native and bits types of the same width.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

#include <cassert>

using namespace loom::adg;

int main() {
  // --- Negative case: mismatched tagged payload widths ---
  {
    ADGBuilder builder("error_tagged_payload_width_mismatch_bad");

    Type tagged32 = Type::tagged(Type::i32(), Type::iN(4));
    Type tagged16 = Type::tagged(Type::i16(), Type::iN(4));

    auto pe32 = builder.newPE("pe32")
        .setLatency(1, 1, 1)
        .setInterval(1, 1, 1)
        .setInterfaceCategory(InterfaceCategory::Tagged)
        .setInputPorts({tagged32, tagged32})
        .setOutputPorts({tagged32})
        .addOp("arith.addi");

    auto pe16 = builder.newPE("pe16")
        .setLatency(1, 1, 1)
        .setInterval(1, 1, 1)
        .setInterfaceCategory(InterfaceCategory::Tagged)
        .setInputPorts({tagged16, tagged16})
        .setOutputPorts({tagged16})
        .addOp("arith.addi");

    auto inst32 = builder.clone(pe32, "pe32_inst");
    auto inst16 = builder.clone(pe16, "pe16_inst");

    auto a = builder.addModuleInput("a", tagged32);
    auto b = builder.addModuleInput("b", tagged32);
    auto c = builder.addModuleInput("c", tagged16);
    auto out = builder.addModuleOutput("out", tagged16);

    builder.connectToModuleInput(a, inst32, 0);
    builder.connectToModuleInput(b, inst32, 1);
    // Connect pe32 output (tagged<i32,i4>) to pe16 input (tagged<i16,i4>):
    // same tag type, different payload widths (32 vs 16).
    builder.connectPorts(inst32, 0, inst16, 0);
    builder.connectToModuleInput(c, inst16, 1);
    builder.connectToModuleOutput(inst16, 0, out);

    auto result = builder.validateADG();
    assert(!result.success && "mismatched tagged payload widths should fail");

    bool foundMismatch = false;
    for (const auto &e : result.errors) {
      if (e.code == "CPL_FABRIC_TYPE_MISMATCH")
        foundMismatch = true;
    }
    assert(foundMismatch && "should report CPL_FABRIC_TYPE_MISMATCH");
  }

  // --- Positive case: matching tagged payload widths ---
  {
    ADGBuilder builder("error_tagged_payload_width_mismatch_ok");

    Type tagged32 = Type::tagged(Type::i32(), Type::iN(4));

    auto pe = builder.newPE("pe32")
        .setLatency(1, 1, 1)
        .setInterval(1, 1, 1)
        .setInterfaceCategory(InterfaceCategory::Tagged)
        .setInputPorts({tagged32, tagged32})
        .setOutputPorts({tagged32})
        .addOp("arith.addi");

    auto inst0 = builder.clone(pe, "pe0");
    auto inst1 = builder.clone(pe, "pe1");

    auto a = builder.addModuleInput("a", tagged32);
    auto b = builder.addModuleInput("b", tagged32);
    auto c = builder.addModuleInput("c", tagged32);
    auto out = builder.addModuleOutput("out", tagged32);

    builder.connectToModuleInput(a, inst0, 0);
    builder.connectToModuleInput(b, inst0, 1);
    builder.connectPorts(inst0, 0, inst1, 0);
    builder.connectToModuleInput(c, inst1, 1);
    builder.connectToModuleOutput(inst1, 0, out);

    auto result = builder.validateADG();
    assert(result.success && "matching tagged payload widths should pass");

    // Export valid MLIR for the test harness.
    builder.exportMLIR(
        "Output/error_tagged_payload_width_mismatch.fabric.mlir");
  }

  return 0;
}
