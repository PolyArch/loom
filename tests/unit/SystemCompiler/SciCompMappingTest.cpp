// Minimal SciComp mapping validation for D6 JSON summary and per-core mapping.

#include "loom/SystemCompiler/InfeasibilityCut.h"
#include "loom/SystemCompiler/KernelCompiler.h"
#include "loom/SystemCompiler/L2CoreCompiler.h"

#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Fabric/FabricDialect.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

struct MappingCandidate {
  std::string sourceFile;
  std::string functionName;
};

struct MappingSuccess {
  std::string coreType;
  std::string sourceFile;
  std::string functionName;
};

static void registerDialects(mlir::MLIRContext &ctx) {
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::math::MathDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx.getOrLoadDialect<mlir::DLTIDialect>();
  ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  ctx.getOrLoadDialect<mlir::ub::UBDialect>();
  ctx.getOrLoadDialect<loom::dataflow::DataflowDialect>();
  ctx.getOrLoadDialect<loom::fabric::FabricDialect>();
  ctx.getOrLoadDialect<circt::handshake::HandshakeDialect>();
}

static std::filesystem::path repoRoot() {
  return std::filesystem::path(LOOM_SOURCE_DIR);
}

static std::string readTextFileSansNul(const std::filesystem::path &path) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path.string());
  if (!bufferOrErr)
    return {};
  llvm::StringRef buffer = (*bufferOrErr)->getBuffer();
  std::string text;
  text.reserve(buffer.size());
  for (char ch : buffer) {
    if (ch != '\0')
      text.push_back(ch);
  }
  return text;
}

static mlir::OwningOpRef<mlir::ModuleOp>
parseModuleFromFile(mlir::MLIRContext &ctx, const std::filesystem::path &path) {
  std::string text = readTextFileSansNul(path);
  if (text.empty())
    return {};
  return mlir::parseSourceString<mlir::ModuleOp>(llvm::StringRef(text), &ctx);
}

static bool loadJsonObject(const std::filesystem::path &path,
                           llvm::json::Object &root) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path.string());
  if (!bufferOrErr)
    return false;
  auto parsed = llvm::json::parse((*bufferOrErr)->getBuffer());
  if (!parsed)
    return false;
  auto *obj = parsed->getAsObject();
  if (!obj)
    return false;
  root = *obj;
  return true;
}

static bool expect(bool cond, const std::string &message) {
  if (cond)
    return true;
  std::cerr << "FAIL: " << message << "\n";
  return false;
}

static std::string cutSummary(const std::optional<loom::InfeasibilityCut> &cut) {
  if (!cut)
    return "none";
  return std::string(loom::cutReasonToString(cut->reason));
}

static bool validateD6Json(const std::filesystem::path &jsonPath) {
  llvm::json::Object root;
  if (!loadJsonObject(jsonPath, root))
    return expect(false, "cannot parse D6_scicomp_arch.json");

  if (!expect(root.getString("name") &&
                  *root.getString("name") == "D6_scicomp",
              "json name must be D6_scicomp")) {
    return false;
  }

  auto *coreTypes = root.getArray("coreTypes");
  if (!expect(coreTypes && coreTypes->size() == 3,
              "json must define exactly 3 core types")) {
    return false;
  }

  struct CoreExpectation {
    const char *typeName;
    int instanceCount;
    int meshRows;
    int meshCols;
    int spmBytes;
  };
  const CoreExpectation expected[] = {
      {"SC-FP", 4, 12, 12, 32 * 1024},
      {"SC-SPM", 3, 8, 8, 64 * 1024},
      {"SC-CTRL", 1, 8, 8, 32 * 1024},
  };

  for (const auto &exp : expected) {
    const llvm::json::Object *found = nullptr;
    for (const auto &entry : *coreTypes) {
      auto *obj = entry.getAsObject();
      if (!obj)
        continue;
      auto typeName = obj->getString("typeName");
      if (typeName && *typeName == exp.typeName) {
        found = obj;
        break;
      }
    }
    if (!expect(found != nullptr,
                std::string("missing core type ") + exp.typeName)) {
      return false;
    }
    if (!expect(found->getInteger("instanceCount") &&
                    *found->getInteger("instanceCount") == exp.instanceCount,
                std::string(exp.typeName) + " instanceCount mismatch")) {
      return false;
    }
    if (!expect(found->getInteger("meshRows") &&
                    *found->getInteger("meshRows") == exp.meshRows,
                std::string(exp.typeName) + " meshRows mismatch")) {
      return false;
    }
    if (!expect(found->getInteger("meshCols") &&
                    *found->getInteger("meshCols") == exp.meshCols,
                std::string(exp.typeName) + " meshCols mismatch")) {
      return false;
    }
    if (!expect(found->getInteger("spmBytes") &&
                    *found->getInteger("spmBytes") == exp.spmBytes,
                std::string(exp.typeName) + " spmBytes mismatch")) {
      return false;
    }
  }

  auto *noc = root.getObject("noc");
  if (!expect(noc != nullptr, "json must define noc")) {
    return false;
  }
  if (!expect(noc->getString("topology") &&
                  *noc->getString("topology") == "mesh",
              "noc topology must be mesh")) {
    return false;
  }
  if (!expect(noc->getInteger("meshRows") &&
                  *noc->getInteger("meshRows") == 2,
              "noc meshRows mismatch")) {
    return false;
  }
  if (!expect(noc->getInteger("meshCols") &&
                  *noc->getInteger("meshCols") == 4,
              "noc meshCols mismatch")) {
    return false;
  }

  auto *sharedMemory = root.getObject("sharedMemory");
  if (!expect(sharedMemory != nullptr, "json must define sharedMemory")) {
    return false;
  }
  if (!expect(sharedMemory->getInteger("l2SizeBytes") &&
                  *sharedMemory->getInteger("l2SizeBytes") == 512 * 1024,
              "sharedMemory l2SizeBytes mismatch")) {
    return false;
  }
  if (!expect(sharedMemory->getInteger("numBanks") &&
                  *sharedMemory->getInteger("numBanks") == 8,
              "sharedMemory numBanks mismatch")) {
    return false;
  }
  if (!expect(sharedMemory->getInteger("bankWidthBytes") &&
                  *sharedMemory->getInteger("bankWidthBytes") == 32,
              "sharedMemory bankWidthBytes mismatch")) {
    return false;
  }

  std::cout << "PASS: validateD6Json\n";
  return true;
}

static std::optional<MappingSuccess>
tryMapFirstCandidate(const std::filesystem::path &root,
                     mlir::MLIRContext &ctx,
                     const std::string &coreType,
                     const std::vector<MappingCandidate> &candidates) {
  const std::filesystem::path kernelsDir =
      root / "tests/e2e/apps/scientific/kernels";
  const std::filesystem::path adgPath =
      root / "adg/scicomp" / (coreType + ".mlir");

  loom::tapestry::KernelCompiler kernelCompiler(ctx, {kernelsDir.string()});
  loom::MapperOptions mapperOpts;
  mapperOpts.budgetSeconds = 120.0;
  mapperOpts.seed = 42;
  mapperOpts.verbose = false;

  for (const auto &candidate : candidates) {
    auto coreModule = parseModuleFromFile(ctx, adgPath);
    if (!coreModule) {
      std::cerr << "FAIL: cannot parse core ADG " << adgPath << "\n";
      return std::nullopt;
    }

    const std::filesystem::path sourcePath = kernelsDir / candidate.sourceFile;
    if (!kernelCompiler.loadSource(sourcePath.string())) {
      std::cerr << "WARN: KernelCompiler loadSource failed for "
                << sourcePath.string() << "\n";
      continue;
    }

    auto kernelResult = kernelCompiler.compile(candidate.functionName);
    if (!kernelResult.success || !kernelResult.dfgModule) {
      std::cerr << "WARN: compile failed for " << candidate.functionName
                << " on " << coreType << ": " << kernelResult.diagnostics
                << "\n";
      continue;
    }

    auto kernelModule = std::move(kernelResult.dfgModule);
    loom::L2Assignment assignment;
    assignment.coreInstanceName = coreType + ".0";
    assignment.coreType = coreType;
    assignment.coreADG = *coreModule;

    loom::L2Assignment::KernelAssignment kernelAssignment;
    kernelAssignment.kernelName = candidate.functionName;
    kernelAssignment.kernelDFG = *kernelModule;
    assignment.kernels.push_back(kernelAssignment);

    loom::L2CoreCompiler l2Compiler;
    loom::L2Result result = l2Compiler.compile(assignment, mapperOpts, &ctx);
    if (result.allKernelsMapped && !result.kernelResults.empty() &&
        result.kernelResults.front().success &&
        !result.costSummary.kernelMetrics.empty() &&
        result.costSummary.kernelMetrics.front().achievedII > 0) {
      std::cout << "PASS: mapped " << coreType << " <- "
                << candidate.functionName << "\n";
      return MappingSuccess{coreType, candidate.sourceFile,
                            candidate.functionName};
    }

    std::cerr << "WARN: mapping failed for " << coreType << " <- "
              << candidate.functionName
              << " cut=" << cutSummary(result.costSummary.cut) << "\n";
  }

  return std::nullopt;
}

static bool testSciCompMappings() {
  const std::filesystem::path root = repoRoot();
  const std::filesystem::path jsonPath =
      root / "adg/scicomp/D6_scicomp_arch.json";
  if (!expect(std::filesystem::exists(jsonPath),
              "missing D6_scicomp_arch.json")) {
    return false;
  }
  if (!validateD6Json(jsonPath))
    return false;

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::DLTIDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::ub::UBDialect>();
  registry.insert<loom::dataflow::DataflowDialect>();
  registry.insert<loom::fabric::FabricDialect>();
  registry.insert<circt::handshake::HandshakeDialect>();

  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();
  registerDialects(ctx);

  const std::vector<MappingCandidate> fpCandidates = {
      {"axpy.c", "axpy_basic"},
  };
  const std::vector<MappingCandidate> spmCandidates = {
      {"spmv.c", "spmv_csr"},
      {"spmv.c", "spmv_ell"},
  };
  const std::vector<MappingCandidate> ctrlCandidates = {
      {"neighbor_rebuild.c", "rebuild_cell_list"},
      {"boundary_apply.c", "boundary_neumann"},
      {"boundary_apply.c", "boundary_dirichlet"},
      {"residual_check.c", "residual_max"},
  };

  auto fpSuccess = tryMapFirstCandidate(root, ctx, "SC-FP", fpCandidates);
  if (!expect(fpSuccess.has_value(), "no SC-FP candidate compiled and mapped")) {
    return false;
  }

  auto spmSuccess = tryMapFirstCandidate(root, ctx, "SC-SPM", spmCandidates);
  if (!expect(spmSuccess.has_value(),
              "no SC-SPM candidate compiled and mapped")) {
    return false;
  }

  auto ctrlSuccess =
      tryMapFirstCandidate(root, ctx, "SC-CTRL", ctrlCandidates);
  if (!expect(ctrlSuccess.has_value(),
              "no SC-CTRL candidate compiled and mapped")) {
    return false;
  }

  std::cout << "PASS: SciCompMappings\n";
  std::cout << "INFO: SC-FP=" << fpSuccess->functionName
            << " SC-SPM=" << spmSuccess->functionName
            << " SC-CTRL=" << ctrlSuccess->functionName << "\n";
  return true;
}

} // namespace

int main() {
  if (!testSciCompMappings())
    return 1;
  return 0;
}
