#include "Frontend/Executable/InstructionCoreBinary.h"

#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Frontend/Executable/CompilerTargetBinding.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(1);
}

void require(llvm::StringRef test, bool condition, llvm::StringRef message) {
  if (!condition)
    fail(test, message.str());
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef marker) {
  if (value)
    fail(test, "accepted a value that must fail closed");
  const std::string message = llvm::toString(value.takeError());
  if (!llvm::StringRef(message).contains(marker))
    fail(test, "expected '" + marker.str() + "' in: " + message);
}

class TemporaryTree final {
public:
  explicit TemporaryTree(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> root;
    std::error_code error = llvm::sys::fs::createUniqueDirectory(
        "loom-instruction-core-binary", root);
    if (error)
      fail(test_, error.message());
    root_ = root.str().str();
    createDirectory("artifacts");
    createDirectory("blobs");
  }

  ~TemporaryTree() {
    std::error_code error = llvm::sys::fs::remove_directories(root_);
    if (error)
      llvm::errs() << "could not remove " << root_ << ": " << error.message()
                   << '\n';
  }

  std::string path(llvm::StringRef leaf) const {
    llvm::SmallString<256> result(root_);
    llvm::sys::path::append(result, leaf);
    return result.str().str();
  }

private:
  void createDirectory(llvm::StringRef leaf) {
    std::error_code error = llvm::sys::fs::create_directory(path(leaf));
    if (error)
      fail(test_, error.message());
  }

  std::string test_;
  std::string root_;
};

mlir::MLIRContext &context() {
  static mlir::MLIRContext *value = [] {
    mlir::DialectRegistry registry;
    registry.insert<dataflow::DataflowDialect, mlir::func::FuncDialect,
                    mlir::arith::ArithDialect, mlir::DLTIDialect,
                    mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
    auto *created =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    created->loadAllAvailableDialects();
    return created;
  }();
  return *value;
}

loom::CompilerTargetPolicy policy() {
  return {loom::fabric::RiscVAbi::Lp64d,
          loom::fabric::RiscVCodeModel::MediumAny,
          loom::fabric::RelocationModel::Static,
          "generic-rv64",
          {}};
}

loom::FinalizedCompilerTargetBinding
targetBinding(llvm::StringRef test, const loom::ArtifactStore &store) {
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  auto cohort = take(test, loom::resolveSystemCompilerTargetBindings(
                               design.roots().front(), policy(), store));
  require(test, !cohort.instructionGroups().empty(),
          "builtin System has no InstructionCore target group");
  return take(
      test,
      loom::importCompilerTargetBinding(
          cohort.instructionGroups().front().binding().reference(), store));
}

struct DataflowFixture final {
  loom::ArtifactRootReference reference;
  std::vector<dataflow::RootThreadLaunchRef> roots;
};

DataflowFixture publishProgram(llvm::StringRef test,
                               const loom::ArtifactStore &store,
                               llvm::StringRef arithmeticOperation) {
  std::string source = R"mlir(
module {
  dataflow.graph private @g(%ctrl: none, %x: i32) -> i32
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %y = )mlir";
  source += arithmeticOperation.str();
  source += R"mlir( %x, %x : i32
    %result:2 = dataflow.sync %ctrl, %y : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(%x: i32)
      ctrl (%ctrl: none) {
    %value, %done = dataflow.graph.launch @g deps(%ctrl) values(%x)
        stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host(%x: i32) {
    %first = dataflow.thread.launch @worker(%x)
        : (i32) -> !dataflow.thread_token
    %second = dataflow.thread.launch @worker(%x)
        : (i32) -> !dataflow.thread_token
    return
  }
}
)mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  require(test, static_cast<bool>(module), "failed to parse Dataflow fixture");
  auto artifact = take(test, dataflow::finalizeCanonicalDataflow(module.get()));
  auto view = take(test, artifact.view());
  std::vector<dataflow::RootThreadLaunchRef> roots;
  for (const auto &launch : view.rootThreadLaunches())
    roots.push_back(launch.ref);
  require(test, roots.size() == 2,
          "Dataflow fixture did not produce two distinct root launches");
  auto reference =
      take(test, dataflow::publishCanonicalDataflow(artifact, store));
  return {std::move(reference), std::move(roots)};
}

void writeBytes(llvm::StringRef test, llvm::StringRef path,
                llvm::ArrayRef<std::uint8_t> bytes) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_None);
  if (error)
    fail(test, error.message());
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  output.close();
  if (output.has_error())
    fail(test, "failed to write " + path.str());
}

std::vector<std::uint8_t>
linkedExecutable(llvm::StringRef test,
                 const loom::CompilerTargetBinding &target,
                 const TemporaryTree &tree) {
  llvm::LLVMContext llvmContext;
  auto module = std::make_unique<llvm::Module>("instruction-core", llvmContext);
  module->setTargetTriple(llvm::Triple(target.targetTriple()));
  module->setDataLayout(target.dataLayout());
  llvm::Function *entry = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(llvmContext), false),
      llvm::GlobalValue::ExternalLinkage, "__loom_thread_entry_0", *module);
  llvm::IRBuilder<> builder(
      llvm::BasicBlock::Create(llvmContext, "entry", entry));
  builder.CreateRetVoid();

  const std::vector<std::uint8_t> object =
      take(test, loom::emitCompilerTargetObject(std::move(module), target));
  const std::string objectPath = tree.path("entry.o");
  const std::string executablePath = tree.path("instruction.elf");
  writeBytes(test, objectPath, object);
  const std::string linker = LOOM_TEST_LLD_PATH;
  const llvm::SmallVector<llvm::StringRef, 12> arguments = {
      linker,
      "-m",
      "elf64lriscv",
      "--entry=__loom_thread_entry_0",
      "-Ttext=0x10000",
      "--no-dynamic-linker",
      "-o",
      executablePath,
      objectPath,
  };
  std::string error;
  bool executionFailed = false;
  const int result = llvm::sys::ExecuteAndWait(
      linker, arguments, std::nullopt, {}, /*SecondsToWait=*/30,
      /*MemoryLimit=*/1024, &error, &executionFailed);
  require(test, !executionFailed && result == 0, "ld.lld failed: " + error);

  auto buffer = llvm::MemoryBuffer::getFile(executablePath, false, false);
  if (!buffer)
    fail(test, buffer.getError().message());
  llvm::StringRef bytes = (*buffer)->getBuffer();
  return std::vector<std::uint8_t>(bytes.bytes_begin(), bytes.bytes_end());
}

loom::InstructionCoreBinaryDraft
draft(const DataflowFixture &dataflow,
      const loom::FinalizedCompilerTargetBinding &target,
      std::vector<std::uint8_t> executable,
      std::vector<loom::ThreadEntryBinding> entries) {
  return {dataflow.reference,
          target.reference(),
          std::move(executable),
          std::move(entries),
          {}};
}

void exactRootedEntriesRoundTrip() {
  const llvm::StringRef test = __func__;
  TemporaryTree tree(test);
  loom::ArtifactStore artifacts(tree.path("artifacts"));
  loom::BlobStore blobs(tree.path("blobs"));
  auto target = targetBinding(test, artifacts);
  DataflowFixture dataflow = publishProgram(test, artifacts, "arith.addi");
  std::vector<std::uint8_t> executable =
      linkedExecutable(test, target.binding(), tree);

  auto finalized =
      take(test, loom::finalizeInstructionCoreBinary(
                     draft(dataflow, target, executable,
                           {{dataflow.roots[1], 0}, {dataflow.roots[0], 0}}),
                     artifacts, blobs));
  auto imported = take(test, loom::importInstructionCoreBinary(
                                 finalized.reference(), artifacts, blobs));
  require(test, imported.binary().threadEntryTable().size() == 2,
          "strict import lost a rooted entry binding");
  require(test,
          take(test, imported.binary().threadEntry(dataflow.roots[0])) == 0 &&
              take(test, imported.binary().threadEntry(dataflow.roots[1])) == 0,
          "two root launches did not share the exact binary entry");
  require(test, !imported.binary().loadSegments().empty(),
          "real executable produced no canonical load segments");
  bool executableSegment = false;
  for (const auto &segment : imported.binary().loadSegments())
    executableSegment |= segment.executable;
  require(test, executableSegment,
          "real executable produced no executable load segment");
  require(test,
          take(test, blobs.get(imported.binary().codeBlob())) == executable,
          "code BlobDigest did not resolve to the exact ELF bytes");
}

void malformedRootRelationsFailClosed() {
  const llvm::StringRef test = __func__;
  TemporaryTree tree(test);
  loom::ArtifactStore artifacts(tree.path("artifacts"));
  loom::BlobStore blobs(tree.path("blobs"));
  auto target = targetBinding(test, artifacts);
  DataflowFixture selected = publishProgram(test, artifacts, "arith.addi");
  DataflowFixture foreign = publishProgram(test, artifacts, "arith.muli");
  const std::vector<std::uint8_t> executable =
      linkedExecutable(test, target.binding(), tree);

  expectError(test,
              loom::finalizeInstructionCoreBinary(
                  draft(selected, target, executable,
                        {{selected.roots[0], 0}, {selected.roots[0], 0}}),
                  artifacts, blobs),
              "instruction_core_binary_duplicate_root");
  expectError(test,
              loom::finalizeInstructionCoreBinary(
                  draft(selected, target, executable, {{foreign.roots[0], 0}}),
                  artifacts, blobs),
              "instruction_core_binary_foreign_root");
  expectError(test,
              loom::finalizeInstructionCoreBinary(
                  draft(selected, target, executable,
                        {{dataflow::RootThreadLaunchRef{
                              selected.reference.artifact,
                              dataflow::RootThreadLaunchId(4096)},
                          0}}),
                  artifacts, blobs),
              "instruction_core_binary_invalid_root");
  expectError(test,
              loom::finalizeInstructionCoreBinary(
                  draft(selected, target, executable, {{selected.roots[0], 1}}),
                  artifacts, blobs),
              "instruction_core_binary_missing_entry");
}

} // namespace

int main() {
  exactRootedEntriesRoundTrip();
  malformedRootRelationsFailClosed();
  return 0;
}
