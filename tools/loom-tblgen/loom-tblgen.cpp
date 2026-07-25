//===- loom-tblgen.cpp - Loom TableGen driver -----------------------------===//
//
// The one generator behind Loom's canonical operation authority. It has no
// backend other than the two registry emitters, so a Loom-specific table can
// only be produced from a declared TableGen record.
//
//===----------------------------------------------------------------------===//

#include "LoomTableGen.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

namespace {

enum ActionType {
  GenOperationSchemas,
  GenImplementationFamilies,
  GenImplementationFamilyEnum,
};

cl::opt<ActionType> action(
    cl::desc("Generator to run:"),
    cl::values(clEnumValN(GenOperationSchemas, "gen-operation-schemas",
                          "Generate the canonical operation schema rows"),
               clEnumValN(GenImplementationFamilies,
                          "gen-implementation-families",
                          "Generate the implementation-family registry rows"),
               clEnumValN(GenImplementationFamilyEnum,
                          "gen-implementation-family-enum",
                          "Generate the implementation-family MLIR enum")));

bool runGenerator(raw_ostream &os, const RecordKeeper &records) {
  switch (action) {
  case GenOperationSchemas:
    loom::tblgen::emitOperationSchemas(records, os);
    return false;
  case GenImplementationFamilies:
    loom::tblgen::emitImplementationFamilies(records, os);
    return false;
  case GenImplementationFamilyEnum:
    loom::tblgen::emitImplementationFamilyEnum(records, os);
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram stackTrace(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &runGenerator);
}
