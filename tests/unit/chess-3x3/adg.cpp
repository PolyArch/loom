//===-- adg.cpp - 3x3 chessboard ADG generator -----------------*- C++ -*-===//

#include "../common/ChessUnitCommon.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("chess-3x3.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "chess-3x3 ADG generator\n");
  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty())
    llvm::sys::fs::create_directories(parentPath);
  llvm::outs() << "Generating chess-3x3 ADG -> " << outputFile << "\n";
  loom::unit::buildChessUnitADG(outputFile, "chess_3x3_test_adg", 3, 3);
  llvm::outs() << "Done.\n";
  return 0;
}
