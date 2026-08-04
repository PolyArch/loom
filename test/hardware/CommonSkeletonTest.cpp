#include "Hardware/RTL/CommonSkeleton.h"

#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
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

FinalizedFabricRoot makeSystemFabric(llvm::StringRef test,
                                     const ArtifactStore &store) {
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  require(test, design.roots().size() == 1,
          "builtin target did not produce one System root");
  return take(test, loom::fabric::importEntireFabricRoot(
                        design.roots().front().reference(), store));
}

FinalizedFabricRoot makeBoundaryOnlyFabric(llvm::StringRef test,
                                           const ArtifactStore &store) {
  mlir::DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
        module {
          fabric.module @passthrough(
              %data: !fabric.bits<32>,
              %tagged: !fabric.bits_tag<4, 5>)
              -> (!fabric.bits<16>, !fabric.bits_tag<0, 3>) {
            fabric.yield %data : !fabric.bits<32> to !fabric.bits<16>,
                         %tagged : !fabric.bits_tag<4, 5>
                             to !fabric.bits_tag<0, 3>
          }
        }
      )mlir",
                                              &context);
  require(test, static_cast<bool>(source),
          "unable to parse boundary-only Fabric fixture");
  ::fabric::ModuleOp root;
  for (::fabric::ModuleOp candidate : source->getOps<::fabric::ModuleOp>()) {
    require(test, !root, "boundary fixture has multiple Module roots");
    root = candidate;
  }
  require(test, static_cast<bool>(root), "boundary fixture has no Module root");
  return take(test, loom::fabric::finalizeFabricRoot(root, store));
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

std::string moduleBoundaryPassthroughBuildsDeterministicSkeleton() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  FinalizedFabricRoot fabric = makeBoundaryOnlyFabric(test, store);

  mlir::MLIRContext firstContext;
  firstContext.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                           circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto first = take(test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                              firstContext, fabric.view()));
  require(test, first.operationLeaves.empty(),
          "boundary-only skeleton invented an operation leaf");
  if (llvm::Error error = loom::hardware::rtl::verifyCommonCirctSkeleton(
          *first.module, fabric.view(), first.operationLeaves))
    fail(test, llvm::toString(std::move(error)));

  mlir::MLIRContext secondContext;
  secondContext.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                            circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto second = take(test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                               secondContext, fabric.view()));
  std::string firstText;
  std::string secondText;
  llvm::raw_string_ostream(firstText) << *first.module;
  llvm::raw_string_ostream(secondText) << *second.module;
  require(test, firstText == secondText,
          "equal Fabric roots produced different CIRCT skeletons");

  const std::string systemVerilog =
      take(test, loom::hardware::rtl::lowerAndExportSpecializedSystemVerilog(
                     *first.module));
  const llvm::StringRef rtl(systemVerilog);
  require(test,
          rtl.contains("input_0_data") && rtl.contains("input_1_tag") &&
              rtl.contains("output_0_data") && rtl.contains("output_1_tag") &&
              rtl.contains("[15:0]") && rtl.contains("[2:0]"),
          "boundary skeleton omitted canonical transport signals");

  FinalizedFabricRoot system = makeSystemFabric(test, store);
  expectError(test,
              loom::hardware::rtl::buildModuleRootCirctSkeleton(secondContext,
                                                                system.view()),
              "requires a Module root");
  return systemVerilog;
}

void writeBoundaryToolArtifacts(const std::filesystem::path &root,
                                llvm::StringRef systemVerilog) {
  std::filesystem::create_directories(root);
  std::ofstream(root / "loom_module.sv") << systemVerilog.str();
  std::ofstream(root / "testbench.sv") << R"sv(
module testbench;
  logic [31:0] input_0_data;
  logic        input_0_valid;
  logic [3:0]  input_1_data;
  logic [4:0]  input_1_tag;
  logic        input_1_valid;
  logic        output_0_ready;
  logic        output_1_ready;
  logic        input_0_ready;
  logic        input_1_ready;
  logic [15:0] output_0_data;
  logic        output_0_valid;
  logic [2:0]  output_1_tag;
  logic        output_1_valid;
  integer      control;

  loom_module dut(.*);

  initial begin
    for (control = 0; control < 16; control = control + 1) begin
      input_0_data = 32'hcafe0000 ^ control;
      input_0_valid = control[3];
      input_1_data = control[3:0];
      input_1_tag = 5'h18 ^ control[4:0];
      input_1_valid = control[2];
      output_0_ready = control[1];
      output_1_ready = control[0];
      #1;
      if (input_0_ready !== output_0_ready ||
          input_1_ready !== output_1_ready ||
          output_0_data !== input_0_data[15:0] ||
          output_0_valid !== input_0_valid ||
          output_1_tag !== input_1_tag[2:0] ||
          output_1_valid !== input_1_valid)
        $fatal(1, "Module boundary passthrough changed transport semantics");
    end
    $finish;
  end
endmodule
)sv";
  std::ofstream(root / "common_skeleton.ys") << R"ys(
read_verilog -sv loom_module.sv
hierarchy -check -top loom_module
check -assert
select -assert-none loom_module/t:$*ff* loom_module/t:$*latch* loom_module/t:$_*FF* loom_module/t:$_*LATCH* loom_module/t:$mem* loom_module/m:*
synth -top loom_module
check -assert
select -assert-none loom_module/t:$*ff* loom_module/t:$*latch* loom_module/t:$_*FF* loom_module/t:$_*LATCH* loom_module/t:$mem* loom_module/m:*
)ys";
}

} // namespace

int main(int argc, char **argv) {
  require("main", argc == 1 || argc == 2,
          "expected at most one output directory");
  commonSkeletonRejectsUnresolvedOrUnboundLeaves();
  const std::string systemVerilog =
      moduleBoundaryPassthroughBuildsDeterministicSkeleton();
  if (argc == 2)
    writeBoundaryToolArtifacts(argv[1], systemVerilog);
  return 0;
}
