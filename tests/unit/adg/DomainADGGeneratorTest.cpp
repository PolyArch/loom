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

#include <iostream>
#include <string>
#include <vector>

using namespace loom::adg;

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

  std::cerr << "\nResults: " << passed << "/" << total << " tests passed\n";
  return (passed == total) ? 0 : 1;
}
