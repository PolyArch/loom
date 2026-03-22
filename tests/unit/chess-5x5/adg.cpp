//===-- adg.cpp - 5x5 chessboard ADG generator -----------------*- C++ -*-===//

#include "../common/ChessUnitCommon.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("chess-5x5.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "chess-5x5 ADG generator\n");
  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty())
    llvm::sys::fs::create_directories(parentPath);
  llvm::outs() << "Generating chess-5x5 ADG -> " << outputFile << "\n";
  loom::unit::buildChessUnitADG(outputFile, "chess_5x5_test_adg", 5, 5);
  llvm::outs() << "Done.\n";
  return 0;
}
