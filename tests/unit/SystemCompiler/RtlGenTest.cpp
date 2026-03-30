// RTL generation wiring test for the config-driven pipeline.
//
// Verifies that multi-core SV generation produces the expected system-level
// collateral without relying on mapper or conversion stages.

#include "loom/SVGen/MultiCoreSVGen.h"
#include "loom/SystemCompiler/ArchitectureFactory.h"

#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Fabric/FabricDialect.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <chrono>
#include <iostream>
#include <string>

using namespace loom;

namespace {

static void registerDialects(mlir::MLIRContext &ctx) {
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::math::MathDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx.getOrLoadDialect<loom::dataflow::DataflowDialect>();
  ctx.getOrLoadDialect<loom::fabric::FabricDialect>();
  ctx.getOrLoadDialect<circt::handshake::HandshakeDialect>();
}

static std::string makeOutputDir() {
  std::error_code ec;
  auto base = std::filesystem::temp_directory_path(ec);
  if (ec)
    return "rtlgen-test-output";
  auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  return (base / ("loom-rtlgen-test-" + std::to_string(stamp))).string();
}

static bool fileExists(const std::string &path) {
  return llvm::sys::fs::exists(path);
}

static std::string readFile(const std::string &path) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr)
    return {};
  return (*bufferOrErr)->getBuffer().str();
}

static bool testMultiCoreSVGeneration() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<loom::dataflow::DataflowDialect>();
  registry.insert<loom::fabric::FabricDialect>();
  registry.insert<circt::handshake::HandshakeDialect>();

  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();
  registerDialects(ctx);

  auto arch = tapestry::buildStandardArchitecture("rtlgen_test", 1, 1, 2, 2,
                                                  ctx);
  if (arch.coreTypes.empty()) {
    std::cerr << "FAIL: architecture build returned no core types\n";
    return false;
  }

  loom::svgen::MultiCoreCompilationDesc compilation;
  compilation.success = true;

  loom::svgen::MultiCoreCoreDesc coreDesc;
  coreDesc.coreInstanceName = "core0";
  coreDesc.coreType = arch.coreTypes[0].name;
  coreDesc.adgModule = arch.coreTypes[0].adgModule;
  compilation.coreDescs.push_back(coreDesc);

  std::string outDir = makeOutputDir();
  std::error_code ec;
  std::filesystem::remove_all(outDir, ec);
  std::filesystem::create_directories(outDir, ec);
  if (ec) {
    std::cerr << "FAIL: cannot create output directory '" << outDir
              << "': " << ec.message() << "\n";
    return false;
  }

  loom::svgen::MultiCoreSVGenOptions opts;
  opts.outputDir = outDir;
  opts.rtlSourceDir = "src/rtl";
  opts.meshRows = 1;
  opts.meshCols = 1;

  auto result = loom::svgen::generateMultiCoreSV(compilation, opts, &ctx);
  if (!result.success) {
    std::cerr << "FAIL: generateMultiCoreSV returned failure\n";
    return false;
  }

  if (!fileExists(result.systemTopFile)) {
    std::cerr << "FAIL: missing system top file: " << result.systemTopFile
              << "\n";
    return false;
  }

  if (!fileExists(result.systemFilelistFile)) {
    std::cerr << "FAIL: missing system filelist: "
              << result.systemFilelistFile << "\n";
    return false;
  }

  if (result.perCoreFilelists.size() != 1) {
    std::cerr << "FAIL: expected 1 per-core filelist, got "
              << result.perCoreFilelists.size() << "\n";
    return false;
  }

  if (!fileExists(result.perCoreFilelists[0])) {
    std::cerr << "FAIL: missing per-core filelist: "
              << result.perCoreFilelists[0] << "\n";
    return false;
  }

  std::string filelist = readFile(result.systemFilelistFile);
  if (filelist.find("src/rtl/design/noc/noc_pkg.sv") == std::string::npos) {
    std::cerr << "FAIL: system filelist does not reference src/rtl\n";
    return false;
  }
  if (filelist.find("tapestry_system_top.sv") == std::string::npos) {
    std::cerr << "FAIL: system filelist does not reference system top\n";
    return false;
  }

  std::cout << "PASS: testMultiCoreSVGeneration\n";
  return true;
}

} // namespace

int main() {
  if (!testMultiCoreSVGeneration())
    return 1;
  return 0;
}
