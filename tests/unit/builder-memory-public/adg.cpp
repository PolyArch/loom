//===-- adg.cpp - Builder public memory helper test ------------*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/ADG/ADGBuilder.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace loom::adg;

static void buildADG(const std::string &outputPath) {
  ADGBuilder builder("builder_memory_public_test_adg");
  constexpr unsigned dataWidth = 64;

  auto fuLoad = builder.defineLoadFU("fu_load", "index", "i32");
  auto fuStore = builder.defineStoreFU("fu_store", "index", "i32");

  auto loadPE =
      builder.defineSingleFUSpatialPE("load_pe", 3, 2, dataWidth, fuLoad);
  auto storePE =
      builder.defineSingleFUSpatialPE("store_pe", 3, 2, dataWidth, fuStore);

  constexpr unsigned numPEs = 2;
  constexpr unsigned peInputs = 3;
  constexpr unsigned peOutputs = 2;
  constexpr unsigned numScalarInputs = 5;
  constexpr unsigned numScalarOutputs = 3;
  constexpr unsigned memSwInputs = 3;
  constexpr unsigned memSwOutputs = 3;

  const unsigned numSwInputs =
      numPEs * peOutputs + memSwInputs + numScalarInputs;
  const unsigned numSwOutputs =
      numPEs * peInputs + memSwOutputs + numScalarOutputs;

  std::vector<unsigned> swInputWidths(numSwInputs, dataWidth);
  std::vector<unsigned> swOutputWidths(numSwOutputs, dataWidth);
  std::vector<std::vector<bool>> fullCrossbar(
      numSwOutputs, std::vector<bool>(numSwInputs, true));
  auto sw = builder.defineSpatialSW("memory_sw", swInputWidths, swOutputWidths,
                                    fullCrossbar);

  MemorySpec memSpec;
  memSpec.name = "mem_0";
  memSpec.ldPorts = 1;
  memSpec.stPorts = 1;
  memSpec.lsqDepth = 0;
  memSpec.memrefType = "memref<256xi32>";
  memSpec.isPrivate = false;
  auto memory = builder.defineMemory(memSpec);

  SwitchBankDomainSpec domain;
  domain.sw = sw;
  domain.numPEs = numPEs;
  domain.peInputCount = peInputs;
  domain.peOutputCount = peOutputs;
  domain.memory = memory;
  domain.numMemories = 1;
  domain.swInputPortsPerMemory = memSwInputs;
  domain.swOutputPortsPerMemory = memSwOutputs;
  domain.scalarInputTypes.assign(numScalarInputs,
                                 "!fabric.bits<" +
                                     std::to_string(dataWidth) + ">");
  domain.scalarOutputTypes.assign(numScalarOutputs,
                                  "!fabric.bits<" +
                                      std::to_string(dataWidth) + ">");

  auto result = builder.buildSwitchBankDomain(
      domain, [&](unsigned idx) { return idx == 0 ? loadPE : storePE; });
  (void)result;

  builder.exportMLIR(outputPath);
}

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("builder-memory-public.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Builder public memory helper ADG generator\n");

  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: " << parentPath
                   << "\n";
      return 1;
    }
  }

  buildADG(outputFile);
  return 0;
}
