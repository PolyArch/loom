#include "Deployment/ExecutableLeaves.h"

#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::deployment;

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
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

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef marker) {
  if (!error)
    fail(test, "accepted invalid input");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(marker), message);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef marker) {
  if (value)
    fail(test, "accepted invalid input");
  expectError(test, value.takeError(), marker);
}

class TemporaryTree final {
public:
  explicit TemporaryTree(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> root;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-deployment-leaves", root))
      fail(test_, error.message());
    root_ = root.str().str();
    std::filesystem::create_directories(path("artifacts"));
    std::filesystem::create_directories(path("blobs"));
  }

  ~TemporaryTree() { std::filesystem::remove_all(root_); }

  std::string path(llvm::StringRef leaf) const {
    llvm::SmallString<256> result(root_);
    llvm::sys::path::append(result, leaf);
    return result.str().str();
  }

private:
  std::string test_;
  std::string root_;
};

mlir::MLIRContext &context() {
  static mlir::MLIRContext *value = [] {
    mlir::DialectRegistry registry;
    registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                    mlir::DLTIDialect, mlir::func::FuncDialect,
                    mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
    auto *created =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    created->loadAllAvailableDialects();
    return created;
  }();
  return *value;
}

CompilerTargetPolicy targetPolicy() {
  return {loom::fabric::RiscVAbi::Lp64d,
          loom::fabric::RiscVCodeModel::MediumAny,
          loom::fabric::RelocationModel::Static,
          "generic-rv64",
          {}};
}

struct TargetFixture final {
  FinalizedCompilerTargetBinding host;
  FinalizedCompilerTargetBinding instruction;
};

TargetFixture targets(llvm::StringRef test, const ArtifactStore &artifacts) {
  auto design = take(test, adg::buildBuiltinTarget(
                               artifacts, adg::BuiltinTargetPreset::Small));
  auto cohort = take(test, resolveSystemCompilerTargetBindings(
                               design.roots().front(), targetPolicy(),
                               artifacts));
  require(test, !cohort.instructionGroups().empty(),
          "builtin System has no InstructionCore target");
  return {take(test, importCompilerTargetBinding(cohort.host().reference(),
                                                 artifacts)),
          take(test, importCompilerTargetBinding(
                         cohort.instructionGroups().front().binding().reference(),
                         artifacts))};
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
    fail(test, "failed to write executable input");
}

std::vector<std::uint8_t>
linkedHostExecutable(llvm::StringRef test,
                     const CompilerTargetBinding &target,
                     const TemporaryTree &tree) {
  llvm::LLVMContext llvmContext;
  auto module = std::make_unique<llvm::Module>("host-program", llvmContext);
  module->setTargetTriple(llvm::Triple(target.targetTriple()));
  module->setDataLayout(target.dataLayout());
  for (llvm::StringRef name : {"loom_host_entry_0", "loom_host_entry_1"}) {
    llvm::Function *entry = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(llvmContext), false),
        llvm::GlobalValue::ExternalLinkage, name, *module);
    llvm::IRBuilder<> builder(
        llvm::BasicBlock::Create(llvmContext, "entry", entry));
    builder.CreateRetVoid();
  }
  const std::vector<std::uint8_t> object =
      take(test, emitCompilerTargetObject(std::move(module), target));
  const std::string objectPath = tree.path("host.o");
  const std::string executablePath = tree.path("host.elf");
  writeBytes(test, objectPath, object);
  const llvm::SmallVector<llvm::StringRef, 12> arguments = {
      LOOM_TEST_LLD_PATH,
      "-m",
      "elf64lriscv",
      "--entry=loom_host_entry_0",
      "-Ttext=0x10000",
      "--no-dynamic-linker",
      "-o",
      executablePath,
      objectPath,
  };
  std::string error;
  bool executionFailed = false;
  const int result = llvm::sys::ExecuteAndWait(
      LOOM_TEST_LLD_PATH, arguments, std::nullopt, {}, 30, 1024, &error,
      &executionFailed);
  require(test, !executionFailed && result == 0, "ld.lld failed: " + error);
  auto buffer = llvm::MemoryBuffer::getFile(executablePath, false, false);
  if (!buffer)
    fail(test, buffer.getError().message());
  return std::vector<std::uint8_t>((*buffer)->getBuffer().bytes_begin(),
                                   (*buffer)->getBuffer().bytes_end());
}

CanonicalTypeBytes typeBytes(llvm::StringRef test, mlir::Type type) {
  auto encoded = take(test, dataflow::encodeCanonicalType(type));
  return CanonicalTypeBytes(encoded.bytes().begin(), encoded.bytes().end());
}

HostProgramLeafDraft hostDraft(llvm::StringRef test,
                               const TargetFixture &target,
                               std::vector<std::uint8_t> programBytes) {
  const CanonicalTypeBytes i32 =
      typeBytes(test, mlir::IntegerType::get(&context(), 32));
  const CanonicalTypeBytes i8 =
      typeBytes(test, mlir::IntegerType::get(&context(), 8));
  const CanonicalTypeBytes memory = typeBytes(
      test, mlir::MemRefType::get({4}, mlir::IntegerType::get(&context(), 32)));
  return {target.host.reference(),
          std::move(programBytes),
          {{1, "loom_host_entry_1", {i8}, {}, {1}},
           {0, "loom_host_entry_0", {i32}, {i32}, {2, 0}}},
          {{2, HostExternalInterfaceKind::Memory,
            HostExternalInterfaceDirection::InOut, memory},
           {0, HostExternalInterfaceKind::Value,
            HostExternalInterfaceDirection::Input, i32},
           {1, HostExternalInterfaceKind::Stream,
            HostExternalInterfaceDirection::Output, i8}},
          {}};
}

struct DataflowFixture final {
  ArtifactRootReference reference;
  dataflow::LogicalMemoryRootRef memory;
};

DataflowFixture publishMemoryProgram(llvm::StringRef test,
                                     const ArtifactStore &artifacts,
                                     llvm::StringRef llvmDataLayout) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @g(%ctrl: none, %mem: memref<4xi32>) -> ()
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return values() streams() memories()
        complete(%ctrl : none)
  }
  dataflow.thread private @t domain(#dataflow.thread_domain<dense>)(
      %mem: memref<4xi32>) ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @g deps(%ctrl) values()
        stream_inputs() memories(%mem) stream_outputs()
        : (none, memref<4xi32>) -> none
    dataflow.thread.yield %done : none
  }
  func.func private @host(%mem: memref<4xi32>) {
    %token = dataflow.thread.launch @t(%mem)
        : (memref<4xi32>) -> !dataflow.thread_token
    return
  }
}
)mlir", &context());
  require(test, static_cast<bool>(module), "failed to parse Dataflow fixture");
  module->getOperation()->setAttr(
      "llvm.data_layout",
      mlir::StringAttr::get(&context(), llvmDataLayout));
  auto finalized = take(test, dataflow::finalizeCanonicalDataflow(*module));
  auto view = take(test, finalized.view());
  require(test, view.logicalMemoryRoots().size() == 1,
          "fixture did not expose one logical memory root");
  const auto memory = view.logicalMemoryRoots().front().ref;
  auto reference = take(test, dataflow::publishCanonicalDataflow(finalized,
                                                                  artifacts));
  return {std::move(reference), memory};
}

void hostLeafCanonicalizesAndBindsRegistration() {
  const llvm::StringRef test = __func__;
  TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  TargetFixture target = targets(test, artifacts);
  HostProgramLeafDraft draft = hostDraft(
      test, target, linkedHostExecutable(test, target.host.binding(), tree));
  HostProgramLeafDraft reordered = draft;
  HostProgramLeafDraft changed = draft;
  std::reverse(reordered.programEntries.begin(), reordered.programEntries.end());
  std::reverse(reordered.externalInterfaces.begin(),
               reordered.externalInterfaces.end());

  HostProgramLeaf first = take(
      test, finalizeHostProgramLeaf(std::move(draft), artifacts, blobs));
  HostProgramLeaf second = take(
      test, finalizeHostProgramLeaf(std::move(reordered), artifacts, blobs));
  require(test, first.programEntries().front().entryOrdinal == 0 &&
                    first.externalInterfaces().front().interfaceOrdinal == 0,
          "host leaf did not canonicalize dense ordinals");
  require(test, first.registrationTableDigest() ==
                    second.registrationTableDigest(),
          "authoring order changed registration identity");
  require(test, take(test, blobs.get(first.programBlob())) ==
                    take(test, blobs.get(second.programBlob())),
          "host program blob changed during finalization");

  changed.externalInterfaces[0].direction =
      HostExternalInterfaceDirection::Input;
  HostProgramLeaf changedLeaf = take(
      test, finalizeHostProgramLeaf(std::move(changed), artifacts, blobs));
  require(test, changedLeaf.registrationTableDigest() !=
                    first.registrationTableDigest(),
          "interface semantics did not change registration digest");
}

void hostLeafRejectsInvalidClosure() {
  const llvm::StringRef test = __func__;
  TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  TargetFixture target = targets(test, artifacts);
  HostProgramLeafDraft valid = hostDraft(
      test, target, linkedHostExecutable(test, target.host.binding(), tree));

  HostProgramLeafDraft duplicate = valid;
  duplicate.programEntries[1].entryOrdinal = 1;
  expectError(test,
              finalizeHostProgramLeaf(std::move(duplicate), artifacts, blobs),
              "duplicate program entry ordinal");

  HostProgramLeafDraft missingInterface = valid;
  missingInterface.programEntries[0].externalInterfaceOrdinals = {19};
  expectError(
      test,
      finalizeHostProgramLeaf(std::move(missingInterface), artifacts, blobs),
      "external interface ordinal is out of range");

  HostProgramLeafDraft malformedType = valid;
  malformedType.externalInterfaces[0].semanticType = {0xff};
  expectError(test,
              finalizeHostProgramLeaf(std::move(malformedType), artifacts,
                                      blobs),
              "semantic type is not canonical");

  HostProgramLeafDraft wrongTarget = valid;
  wrongTarget.compilerTargetBinding = target.instruction.reference();
  expectError(test,
              finalizeHostProgramLeaf(std::move(wrongTarget), artifacts,
                                      blobs),
              "requires a HostCore");
}

frontend::StaticGlobalMemoryCatalog
catalog(const CompilerTargetBinding &target, std::vector<std::uint8_t> bytes) {
  return {target.dataLayout().str(),
          {{"table", frontend::StaticGlobalProvision::Image,
            frontend::StaticMemoryPermissions::ReadOnly, 16, 16,
            std::move(bytes)}}};
}

void staticMemoryLeafUsesExactLogicalRoot() {
  const llvm::StringRef test = __func__;
  TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  TargetFixture target = targets(test, artifacts);
  DataflowFixture dataflow = publishMemoryProgram(
      test, artifacts, target.host.binding().dataLayout());

  auto initialized = catalog(target.host.binding(),
                             {0, 1, 2, 3, 4, 5, 6, 7,
                              8, 9, 10, 11, 12, 13, 14, 15});
  StaticMemoryImageLeaf image = take(
      test, buildStaticMemoryImageLeaf(
                dataflow.reference, dataflow.memory, target.host.reference(),
                initialized, 0, artifacts, blobs));
  require(test, image.initializedChunks().size() == 1 &&
                    image.zeroFillRanges().empty() &&
                    image.sizeBytes() == 16 && image.alignmentBytes() == 16,
          "initialized image did not preserve the exact global layout");
  require(test,
          take(test, blobs.get(image.initializedChunks().front().blobDigest)) ==
              initialized.globals.front().bytes,
          "initialized chunk does not reference exact LLVM bytes");

  auto zero = catalog(target.host.binding(), std::vector<std::uint8_t>(16, 0));
  StaticMemoryImageLeaf zeroImage = take(
      test, buildStaticMemoryImageLeaf(
                dataflow.reference, dataflow.memory, target.host.reference(),
                zero, 0, artifacts, blobs));
  require(test, zeroImage.initializedChunks().empty() &&
                    zeroImage.zeroFillRanges().size() == 1,
          "zero initializer was not represented as zero fill");

  auto wrongLayout = initialized;
  wrongLayout.dataLayout = "e-p:32:32";
  expectError(test,
              buildStaticMemoryImageLeaf(
                  dataflow.reference, dataflow.memory, target.host.reference(),
                  wrongLayout, 0, artifacts, blobs),
              "DataLayout is not structurally compatible");

  auto wrongExtent = initialized;
  wrongExtent.globals.front().sizeBytes = 15;
  wrongExtent.globals.front().bytes.resize(15);
  expectError(test,
              buildStaticMemoryImageLeaf(
                  dataflow.reference, dataflow.memory, target.host.reference(),
                  wrongExtent, 0, artifacts, blobs),
              "logical memory extent");

  DataflowFixture wrongDataflow =
      publishMemoryProgram(test, artifacts, "e-p:32:32");
  expectError(test,
              buildStaticMemoryImageLeaf(
                  wrongDataflow.reference, wrongDataflow.memory,
                  target.host.reference(), initialized, 0, artifacts, blobs),
              "DataLayout is not structurally compatible");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one test mode");
  const llvm::StringRef mode(argv[1]);
  if (mode == "host-canonical")
    hostLeafCanonicalizesAndBindsRegistration();
  else if (mode == "host-invalid")
    hostLeafRejectsInvalidClosure();
  else if (mode == "static-memory")
    staticMemoryLeafUsesExactLogicalRoot();
  else
    fail("main", "unknown test mode: " + mode.str());
  return EXIT_SUCCESS;
}
