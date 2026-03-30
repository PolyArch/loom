//===-- DomainADGGeneratorTest.cpp - DomainADGGenerator unit tests --*- C++ -*-//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the DomainADGGenerator: verifies that D1-D6 domain-specific
// types produce non-empty, parseable Fabric MLIR ADGs via the ADGBuilder API.
//
//===----------------------------------------------------------------------===//

#include "loom/ADG/DomainADGGenerator.h"
#include "loom/SVGen/SVModuleRegistry.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"

#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Fabric/FabricDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/MemoryBuffer.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace loom::adg;

static unsigned countSubstring(const std::string &text,
                               const std::string &needle) {
  unsigned count = 0;
  std::size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

static void registerSciCompDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  loom::dataflow::DataflowDialect, loom::fabric::FabricDialect,
                  circt::handshake::HandshakeDialect>();
}

static std::optional<std::string>
extractFirstPEBlock(const std::string &text) {
  size_t start = std::string::npos;
  size_t spatial = text.find("  fabric.spatial_pe @");
  size_t temporal = text.find("  fabric.temporal_pe @");
  if (spatial != std::string::npos)
    start = spatial;
  if (temporal != std::string::npos &&
      (start == std::string::npos || temporal < start))
    start = temporal;
  if (start == std::string::npos)
    return std::nullopt;

  int braceDepth = 0;
  bool sawOpenBrace = false;
  size_t end = start;
  for (; end < text.size(); ++end) {
    char ch = text[end];
    if (ch == '{') {
      ++braceDepth;
      sawOpenBrace = true;
      continue;
    }
    if (ch == '}') {
      if (braceDepth > 0)
        --braceDepth;
      if (sawOpenBrace && braceDepth == 0) {
        ++end;
        break;
      }
    }
  }
  if (!sawOpenBrace || braceDepth != 0)
    return std::nullopt;

  std::string scaffold = "module {\n";
  scaffold.append(text.substr(start, end - start));
  scaffold.append("\n}\n");
  return scaffold;
}

static bool parseModuleText(const std::string &text) {
  auto scaffold = extractFirstPEBlock(text);
  if (!scaffold)
    return false;
  mlir::DialectRegistry registry;
  registerSciCompDialects(registry);
  mlir::MLIRContext ctx(registry);
  return static_cast<bool>(
      mlir::parseSourceString<mlir::ModuleOp>(llvm::StringRef(*scaffold), &ctx));
}

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

/// Test 1: isValidDomainTypeId correctly accepts D1-D6 and rejects others.
static bool testValidation() {
  bool ok = true;

  // Valid IDs
  for (int i = 1; i <= 6; ++i) {
    std::string id = "D" + std::to_string(i);
    if (!isValidDomainTypeId(id)) {
      std::cerr << "FAIL: testValidation - " << id
                << " should be valid\n";
      ok = false;
    }
  }

  // Invalid IDs
  std::vector<std::string> invalid = {
      "D0", "D7", "D10", "D", "CISY8", "d1", "", "X1"};
  for (const auto &id : invalid) {
    if (isValidDomainTypeId(id)) {
      std::cerr << "FAIL: testValidation - \"" << id
                << "\" should be invalid\n";
      ok = false;
    }
  }

  if (ok)
    std::cerr << "PASS: testValidation\n";
  return ok;
}

/// Test 2: domainParamsFromTypeId returns correct parameters for D1.
static bool testD1Params() {
  DomainTypeParams p = domainParamsFromTypeId("D1");
  bool ok = true;

  if (p.typeId != "D1") {
    std::cerr << "FAIL: testD1Params - typeId=" << p.typeId << "\n";
    ok = false;
  }
  if (p.arrayRows != 6 || p.arrayCols != 6) {
    std::cerr << "FAIL: testD1Params - array=" << p.arrayRows << "x"
              << p.arrayCols << "\n";
    ok = false;
  }
  if (p.fuFpCount != 4) {
    std::cerr << "FAIL: testD1Params - fuFpCount=" << p.fuFpCount << "\n";
    ok = false;
  }
  if (p.spmSizeKB != 64) {
    std::cerr << "FAIL: testD1Params - spmSizeKB=" << p.spmSizeKB << "\n";
    ok = false;
  }
  if (p.isTemporal) {
    std::cerr << "FAIL: testD1Params - should be spatial\n";
    ok = false;
  }
  if (p.totalPEs() != 36) {
    std::cerr << "FAIL: testD1Params - totalPEs=" << p.totalPEs() << "\n";
    ok = false;
  }

  if (ok)
    std::cerr << "PASS: testD1Params\n";
  return ok;
}

/// Test 3: D3 is temporal with correct instruction slots and registers.
static bool testD3Temporal() {
  DomainTypeParams p = domainParamsFromTypeId("D3");
  bool ok = true;

  if (!p.isTemporal) {
    std::cerr << "FAIL: testD3Temporal - should be temporal\n";
    ok = false;
  }
  if (p.instructionSlots != 8) {
    std::cerr << "FAIL: testD3Temporal - slots=" << p.instructionSlots << "\n";
    ok = false;
  }
  if (p.numRegisters != 8) {
    std::cerr << "FAIL: testD3Temporal - regs=" << p.numRegisters << "\n";
    ok = false;
  }

  if (ok)
    std::cerr << "PASS: testD3Temporal\n";
  return ok;
}

/// Test 4: D4 uses 64-bit data width and has no FP FUs.
static bool testD4CryptoWidth() {
  DomainTypeParams p = domainParamsFromTypeId("D4");
  bool ok = true;

  if (p.dataWidth != 64) {
    std::cerr << "FAIL: testD4CryptoWidth - dataWidth=" << p.dataWidth << "\n";
    ok = false;
  }
  if (p.hasFP()) {
    std::cerr << "FAIL: testD4CryptoWidth - should not have FP\n";
    ok = false;
  }

  if (ok)
    std::cerr << "PASS: testD4CryptoWidth\n";
  return ok;
}

/// Test 5: allDomainTypeIds returns exactly 6 IDs in canonical order.
static bool testEnumeration() {
  auto ids = allDomainTypeIds();
  bool ok = true;

  if (ids.size() != 6) {
    std::cerr << "FAIL: testEnumeration - size=" << ids.size() << "\n";
    ok = false;
  }
  for (unsigned i = 0; i < ids.size(); ++i) {
    std::string expected = "D" + std::to_string(i + 1);
    if (ids[i] != expected) {
      std::cerr << "FAIL: testEnumeration - ids[" << i << "]=" << ids[i]
                << " expected " << expected << "\n";
      ok = false;
    }
  }

  if (ok)
    std::cerr << "PASS: testEnumeration\n";
  return ok;
}

/// Test 6: generateDomainADG produces non-empty MLIR for all 6 types.
static bool testGenerateAllTypes() {
  bool ok = true;

  for (const auto &params : allDomainTypes()) {
    std::string mlir = generateDomainADG(params);
    if (mlir.empty()) {
      std::cerr << "FAIL: testGenerateAllTypes - " << params.typeId
                << " produced empty MLIR\n";
      ok = false;
      continue;
    }

    // Verify the generated MLIR contains the expected module name
    std::string expectedModName = params.typeId + "_core";
    if (mlir.find(expectedModName) == std::string::npos) {
      std::cerr << "FAIL: testGenerateAllTypes - " << params.typeId
                << " missing module name \"" << expectedModName << "\"\n";
      ok = false;
    }

    // Verify it contains fabric.module (not just "module @...")
    if (mlir.find("fabric.module") == std::string::npos) {
      std::cerr << "FAIL: testGenerateAllTypes - " << params.typeId
                << " missing fabric.module\n";
      ok = false;
    }

    // Verify PEs are present
    bool hasPE = (mlir.find("spatial_pe") != std::string::npos ||
                  mlir.find("temporal_pe") != std::string::npos);
    if (!hasPE) {
      std::cerr << "FAIL: testGenerateAllTypes - " << params.typeId
                << " missing PE definitions\n";
      ok = false;
    }
  }

  if (ok)
    std::cerr << "PASS: testGenerateAllTypes\n";
  return ok;
}

/// Test 7: D1 (spatial) generates spatial_pe, D3 (temporal) generates
/// temporal_pe.
static bool testPEType() {
  bool ok = true;

  // D1 is spatial
  std::string d1mlir = generateDomainADG(domainParamsFromTypeId("D1"));
  if (d1mlir.find("spatial_pe") == std::string::npos) {
    std::cerr << "FAIL: testPEType - D1 should use spatial_pe\n";
    ok = false;
  }

  // D3 is temporal
  std::string d3mlir = generateDomainADG(domainParamsFromTypeId("D3"));
  if (d3mlir.find("temporal_pe") == std::string::npos) {
    std::cerr << "FAIL: testPEType - D3 should use temporal_pe\n";
    ok = false;
  }

  // D5 is temporal
  std::string d5mlir = generateDomainADG(domainParamsFromTypeId("D5"));
  if (d5mlir.find("temporal_pe") == std::string::npos) {
    std::cerr << "FAIL: testPEType - D5 should use temporal_pe\n";
    ok = false;
  }

  if (ok)
    std::cerr << "PASS: testPEType\n";
  return ok;
}

/// Test 8: D4 (64-bit) generates i64-typed FUs.
static bool testD4Wide() {
  std::string mlir = generateDomainADG(domainParamsFromTypeId("D4"));
  bool ok = true;

  if (mlir.find("i64") == std::string::npos) {
    std::cerr << "FAIL: testD4Wide - D4 should reference i64 types\n";
    ok = false;
  }

  // D4 has fu_fp_count=0, so there should be no FP operations
  if (mlir.find("arith.addf") != std::string::npos) {
    std::cerr << "FAIL: testD4Wide - D4 should not have arith.addf\n";
    ok = false;
  }

  if (ok)
    std::cerr << "PASS: testD4Wide\n";
  return ok;
}

/// Test 9: Invalid type ID returns empty params with empty typeId.
static bool testInvalidTypeId() {
  DomainTypeParams p = domainParamsFromTypeId("INVALID");
  bool ok = true;

  if (!p.typeId.empty()) {
    std::cerr << "FAIL: testInvalidTypeId - typeId should be empty, got \""
              << p.typeId << "\"\n";
    ok = false;
  }

  if (ok)
    std::cerr << "PASS: testInvalidTypeId\n";
  return ok;
}

/// Test 10: SciComp type validation and parameter decoding.
static bool testSciCompParams() {
  bool ok = true;

  if (!isValidSciCompTypeId("SC-FP") || !isValidSciCompTypeId("SC-SPM") ||
      !isValidSciCompTypeId("SC-CTRL") || isValidSciCompTypeId("SC-XYZ")) {
    std::cerr << "FAIL: testSciCompParams - validation mismatch\n";
    ok = false;
  }

  SciCompTypeParams fp = sciCompParamsFromTypeId("SC-FP");
  SciCompTypeParams spm = sciCompParamsFromTypeId("SC-SPM");
  SciCompTypeParams ctrl = sciCompParamsFromTypeId("SC-CTRL");

  if (fp.totalPEs() != 144 || fp.dataWidth != 64 || !fp.decomposable ||
      fp.subLaneBits != 32 || fp.totalFPUnits() != 15 || !fp.hasFMA ||
      !fp.hasRSQRT || !fp.hasFPMin) {
    std::cerr << "FAIL: testSciCompParams - SC-FP params mismatch\n";
    ok = false;
  }
  if (spm.totalPEs() != 64 || spm.spmSizeKB != 64 || spm.spmLdPorts != 4 ||
      spm.extMemLdPorts != 4 || !spm.hasIndirectLoad) {
    std::cerr << "FAIL: testSciCompParams - SC-SPM params mismatch\n";
    ok = false;
  }
  if (!ctrl.isTemporal || ctrl.dataWidth != 32 || ctrl.instructionSlots != 16 ||
      ctrl.numRegisters != 8 || ctrl.operandBufferSize != 4 ||
      !ctrl.hasScatterStore || !ctrl.hasBranch) {
    std::cerr << "FAIL: testSciCompParams - SC-CTRL params mismatch\n";
    ok = false;
  }

  auto ids = allSciCompTypeIds();
  if (ids.size() != 3 || ids[0] != "SC-FP" || ids[1] != "SC-SPM" ||
      ids[2] != "SC-CTRL") {
    std::cerr << "FAIL: testSciCompParams - enumeration mismatch\n";
    ok = false;
  }

  if (ok)
    std::cerr << "PASS: testSciCompParams\n";
  return ok;
}

/// Test 11: SciComp ADGs encode the required structure and capabilities.
static bool testGenerateSciCompTypes() {
  bool ok = true;

  std::string scfp = generateSciCompADG(sciCompParamsFromTypeId("SC-FP"));
  std::string scspm = generateSciCompADG(sciCompParamsFromTypeId("SC-SPM"));
  std::string scctrl = generateSciCompADG(sciCompParamsFromTypeId("SC-CTRL"));

  auto require = [&](const std::string &mlir, const std::string &needle,
                     const char *msg) {
    if (mlir.find(needle) == std::string::npos) {
      std::cerr << "FAIL: testGenerateSciCompTypes - missing " << msg << "\n";
      ok = false;
    }
  };

  require(scfp, "loom.scicomp_khg_type = \"SC-FP\"", "SC-FP type attr");
  require(scfp, "loom.decomposable = true", "SC-FP decomposable attr");
  require(scfp, "loom.sub_lane_bits = 32", "SC-FP sub-lane attr");
  require(scfp, "loom.routing_topology = \"CHESS\"", "SC-FP topology attr");
  require(scfp, "math.fma", "SC-FP FMA op");
  require(scfp, "math.rsqrt", "SC-FP RSQRT op");
  require(scfp, "arith.minimumf", "SC-FP MIN op");
  require(scfp, "f64", "SC-FP f64 types");

  if (countSubstring(scfp, "fabric.instance @SC_FP_core_spe") != 144) {
    std::cerr << "FAIL: testGenerateSciCompTypes - SC-FP expected 144 PEs\n";
    ok = false;
  }

  require(scspm, "loom.scicomp_khg_type = \"SC-SPM\"", "SC-SPM type attr");
  require(scspm, "loom.spm_ld_ports = 4", "SC-SPM SPM ld attr");
  require(scspm, "loom.extmem_ld_ports = 4", "SC-SPM extmem ld attr");
  require(scspm, "loom.has_indirect_load = true", "SC-SPM indirect attr");
  require(scspm, "SC_SPM_core_indirect_load", "SC-SPM indirect FU");
  require(scspm, "f64", "SC-SPM f64 types");
  if (countSubstring(scspm, "fabric.instance @SC_SPM_core_spe") != 64) {
    std::cerr << "FAIL: testGenerateSciCompTypes - SC-SPM expected 64 PEs\n";
    ok = false;
  }

  require(scctrl, "loom.scicomp_khg_type = \"SC-CTRL\"",
          "SC-CTRL type attr");
  require(scctrl, "loom.routing_topology = \"MESH\"", "SC-CTRL topology attr");
  require(scctrl, "loom.operand_buffer_size = 4", "SC-CTRL operand buffer");
  require(scctrl, "temporal_pe", "SC-CTRL temporal PE");
  require(scctrl, "num_instruction = 16", "SC-CTRL instruction slots");
  require(scctrl, "num_register = 8", "SC-CTRL register count");
  require(scctrl, "operand_buffer_size = 4", "SC-CTRL PE operand buffer");
  require(scctrl, "SC_CTRL_core_scatter_store", "SC-CTRL scatter store FU");
  require(scctrl, "SC_CTRL_core_branch", "SC-CTRL branch FU");
  if (countSubstring(scctrl, "fabric.instance @SC_CTRL_core_tpe") != 64) {
    std::cerr << "FAIL: testGenerateSciCompTypes - SC-CTRL expected 64 PEs\n";
    ok = false;
  }

  if (!parseModuleText(scfp) || !parseModuleText(scspm) ||
      !parseModuleText(scctrl)) {
    std::cerr << "FAIL: testGenerateSciCompTypes - generated SciComp ADG did "
                 "not parse\n";
    ok = false;
  }

  if (ok)
    std::cerr << "PASS: testGenerateSciCompTypes\n";
  return ok;
}

/// Test 12: SV registry recognizes the SciComp-specific body ops.
static bool testSciCompSVRegistry() {
  loom::svgen::SVModuleRegistry registry;
  bool ok = true;

  auto require = [&](llvm::StringRef opName, llvm::StringRef expectedPath) {
    if (!loom::svgen::SVModuleRegistry::isKnownOp(opName)) {
      std::cerr << "FAIL: testSciCompSVRegistry - missing op "
                << opName.str() << "\n";
      ok = false;
      return;
    }
    if (loom::svgen::SVModuleRegistry::getSVFilePath(opName) != expectedPath) {
      std::cerr << "FAIL: testSciCompSVRegistry - bad path for "
                << opName.str() << "\n";
      ok = false;
    }
    if (!registry.requireArithOp(opName, "")) {
      std::cerr << "FAIL: testSciCompSVRegistry - requireArithOp rejected "
                << opName.str() << "\n";
      ok = false;
    }
  };

  require("math.rsqrt", "math/fu_op_rsqrt.sv");
  require("arith.minimumf", "arith/fu_op_minimumf.sv");

  if (ok)
    std::cerr << "PASS: testSciCompSVRegistry\n";
  return ok;
}

/// Test 13: export SciComp ADGs to the canonical repo paths.
static bool testExportSciCompTypes() {
  bool ok = true;
  std::vector<std::string> paths = {
      "adg/scicomp/SC-FP.mlir",
      "adg/scicomp/SC-SPM.mlir",
      "adg/scicomp/SC-CTRL.mlir",
  };

  for (const auto &path : paths)
    std::remove(path.c_str());

  for (const auto &params : allSciCompTypes())
    exportSciCompADG(params, "adg/scicomp/" + params.typeId + ".mlir");

  for (const auto &path : paths) {
    auto buf = llvm::MemoryBuffer::getFile(path);
    if (!buf) {
      std::cerr << "FAIL: testExportSciCompTypes - missing file " << path
                << "\n";
      ok = false;
      continue;
    }
    std::string contents = (*buf)->getBuffer().str();
    if (contents.find("fabric.module") == std::string::npos ||
        contents.find("loom.scicomp_khg_type") == std::string::npos) {
      std::cerr << "FAIL: testExportSciCompTypes - malformed file " << path
                << "\n";
      ok = false;
    }
    if (contents.find("@SC-") != std::string::npos) {
      std::cerr << "FAIL: testExportSciCompTypes - illegal symbol name in "
                << path << "\n";
      ok = false;
    }
    if (!parseModuleText(contents)) {
      std::cerr << "FAIL: testExportSciCompTypes - cannot parse " << path
                << "\n";
      ok = false;
      continue;
    }
  }

  if (ok)
    std::cerr << "PASS: testExportSciCompTypes\n";
  return ok;
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main() {
  int passed = 0;
  int total = 0;

  auto run = [&](bool (*test)()) {
    total++;
    if (test())
      passed++;
  };

  run(testValidation);
  run(testD1Params);
  run(testD3Temporal);
  run(testD4CryptoWidth);
  run(testEnumeration);
  run(testGenerateAllTypes);
  run(testPEType);
  run(testD4Wide);
  run(testInvalidTypeId);
  run(testSciCompParams);
  run(testGenerateSciCompTypes);
  run(testSciCompSVRegistry);
  run(testExportSciCompTypes);

  std::cerr << "\nResults: " << passed << "/" << total << " tests passed\n";
  return (passed == total) ? 0 : 1;
}
