//===-- adg.cpp - Builder switch bank helper test ---------------*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//

#include "fcc/ADG/ADGBuilder.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace fcc::adg;

static void buildADG(const std::string &outputPath) {
  ADGBuilder builder("builder_switch_bank_test_adg");
  constexpr unsigned dataWidth = 64;
  constexpr unsigned peInputs = 2;
  constexpr unsigned peOutputs = 1;

  auto fuAdd = builder.defineBinaryFU("fu_add", "arith.addi", "i32", "i32");
  auto pe = builder.defineSingleFUSpatialPE("add_pe", peInputs, peOutputs,
                                            dataWidth, fuAdd);

  constexpr unsigned numPEs = 2;
  constexpr unsigned numScalarInputs = 3;
  constexpr unsigned numScalarOutputs = 1;
  std::vector<unsigned> swInputWidths(numPEs * peOutputs + numScalarInputs,
                                      dataWidth);
  std::vector<unsigned> swOutputWidths(numPEs * peInputs + numScalarOutputs,
                                       dataWidth);
  std::vector<std::vector<bool>> fullCrossbar(
      swOutputWidths.size(),
      std::vector<bool>(swInputWidths.size(), true));
  auto sw = builder.defineSpatialSW("bank_sw", swInputWidths, swOutputWidths,
                                    fullCrossbar);

  SwitchBankDomainSpec domain;
  domain.sw = sw;
  domain.pe = pe;
  domain.numPEs = numPEs;
  domain.peInputCount = peInputs;
  domain.peOutputCount = peOutputs;
  domain.scalarInputTypes.assign(numScalarInputs,
                                 "!fabric.bits<" +
                                     std::to_string(dataWidth) + ">");
  domain.scalarOutputTypes.assign(numScalarOutputs,
                                  "!fabric.bits<" +
                                      std::to_string(dataWidth) + ">");
  (void)builder.buildSwitchBankDomain(domain);

  builder.exportMLIR(outputPath);
}

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("builder-switch-bank.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Builder switch bank helper ADG generator\n");

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
