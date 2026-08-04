#include "Hardware/RTL/CommonSkeleton.h"

#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::fabric::FabricFuOccurrenceNodeRef;
using loom::fabric::FinalizedFabricRoot;
using loom::hardware::rtl::FabricOperationLeafAssociation;

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
                 llvm::StringRef expected) {
  if (!error)
    fail(test, "accepted invalid common CIRCT skeleton");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected), message);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted invalid common CIRCT skeleton");
  expectError(test, value.takeError(), expected);
}

class TemporaryDirectory final {
public:
  explicit TemporaryDirectory(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-common-skeleton-test", path))
      fail(test, error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << test_ << ": unable to remove temporary directory: "
                   << error.message() << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string test_;
  std::string path_;
};

FinalizedFabricRoot makeFabric(llvm::StringRef test,
                               const ArtifactStore &store) {
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  require(test, design.roots().size() == 1,
          "builtin target did not produce one System root");
  const auto dependencies = design.roots().front().directDependencies();
  require(test, dependencies.size() == 1,
          "builtin System root did not name one module dependency");
  return take(test, loom::fabric::importEntireFabricRoot(
                        dependencies.front().root, store));
}

FabricFuOccurrenceNodeRef
findOperationOccurrence(llvm::StringRef test,
                        const loom::fabric::FabricArtifactView &view) {
  for (const auto occurrence : view.fuOccurrences()) {
    const auto definition = view.fuTemplateOf(occurrence);
    if (!definition)
      continue;
    const auto capabilities = view.resolvedFabricOpCapabilities(*definition);
    if (capabilities.empty())
      continue;
    return take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                          view, capabilities.front().occurrence, occurrence));
  }
  fail(test, "builtin Fabric has no concrete operation occurrence");
}

void commonSkeletonRejectsUnresolvedOrUnboundLeaves() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  FinalizedFabricRoot fabric = makeFabric(test, store);
  const FabricFuOccurrenceNodeRef occurrence =
      findOperationOccurrence(test, fabric.view());

  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());

  auto schema = circt::hw::HWGeneratorSchemaOp::create(
      builder, location,
      loom::hardware::rtl::fabricOperationGeneratorSchemaSymbol,
      loom::hardware::rtl::fabricOperationGeneratorDescriptor,
      builder.getArrayAttr({}));
  auto leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(
          &context, loom::hardware::rtl::fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("loom_fabric_operation_0"),
      llvm::ArrayRef<circt::hw::PortInfo>{});
  circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("loom_common_skeleton_test"),
      circt::hw::ModulePortInfo({}, {}),
      [](mlir::OpBuilder &, circt::hw::HWModulePortAccessor &) {});

  const std::vector<FabricOperationLeafAssociation> association = {
      {leaf, occurrence}};
  if (llvm::Error error = loom::hardware::rtl::verifyCommonCirctSkeleton(
          *module, fabric.view(), association))
    fail(test, llvm::toString(std::move(error)));

  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(*module,
                                                             fabric.view(), {}),
              "has no exact Fabric occurrence association");
  const std::vector<FabricOperationLeafAssociation> duplicate = {
      {leaf, occurrence}, {leaf, occurrence}};
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.view(), duplicate),
              "associated more than once");

  auto secondLeaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(
          &context, loom::hardware::rtl::fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("loom_fabric_operation_1"),
      llvm::ArrayRef<circt::hw::PortInfo>{});
  const std::vector<FabricOperationLeafAssociation> duplicateOccurrence = {
      {leaf, occurrence}, {secondLeaf, occurrence}};
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.view(), duplicateOccurrence),
              "occurrence is associated more than once");
  secondLeaf.erase();

  mlir::OwningOpRef<mlir::ModuleOp> foreignModule =
      mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(foreignModule->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location,
      loom::hardware::rtl::fabricOperationGeneratorSchemaSymbol,
      loom::hardware::rtl::fabricOperationGeneratorDescriptor,
      builder.getArrayAttr({}));
  auto foreignLeaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(
          &context, loom::hardware::rtl::fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("foreign_fabric_operation"),
      llvm::ArrayRef<circt::hw::PortInfo>{});
  const std::vector<FabricOperationLeafAssociation> foreignAssociation = {
      {foreignLeaf, occurrence}};
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.view(), foreignAssociation),
              "does not name a Loom leaf in this module");

  schema.setDescriptor("unexpected.fabric.operation");
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.view(), association),
              "schema has an unexpected descriptor");
  schema.setDescriptor(loom::hardware::rtl::fabricOperationGeneratorDescriptor);

  FabricFuOccurrenceNodeRef foreign = occurrence;
  foreign.ordinal += 1000000;
  const std::vector<FabricOperationLeafAssociation> invalid = {{leaf, foreign}};
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.view(), invalid),
              "does not resolve to a concrete Fabric operation capability");

  expectError(
      test,
      loom::hardware::rtl::lowerAndExportSpecializedSystemVerilog(*module),
      "unresolved Loom Fabric operation leaf");

  leaf.erase();
  const std::string systemVerilog = take(
      test,
      loom::hardware::rtl::lowerAndExportSpecializedSystemVerilog(*module));
  require(test,
          llvm::StringRef(systemVerilog)
              .contains("module loom_common_skeleton_test"),
          "specialized CIRCT module did not export SystemVerilog");
}

} // namespace

int main() {
  commonSkeletonRejectsUnresolvedOrUnboundLeaves();
  return 0;
}
