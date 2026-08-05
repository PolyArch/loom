#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/OperationLeaf.h"

#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
using loom::hardware::FinalizedConfigurationABI;
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
  require(
      test, llvm::StringRef(message).contains(expected),
      (llvm::Twine("expected '") + expected + "', received '" + message + "'")
          .str());
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted invalid common CIRCT skeleton");
  expectError(test, value.takeError(), expected);
}

template <typename T>
void expectStructuralUnsupported(llvm::StringRef test,
                                 llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted unsupported Fabric structural topology");
  std::string reason;
  std::string unexpected;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const loom::hardware::rtl::FabricStructuralLoweringUnsupportedError
              &error) { reason = error.reason().str(); },
      [&](const llvm::ErrorInfoBase &error) {
        llvm::raw_string_ostream stream(unexpected);
        error.log(stream);
      });
  require(test, unexpected.empty(),
          "unsupported topology returned the wrong typed error: " + unexpected);
  require(test, !reason.empty(), "unsupported topology has no diagnostic");
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

FinalizedFabricRoot makeOperationFabric(llvm::StringRef test,
                                        const ArtifactStore &store,
                                        bool twoOccurrences = false) {
  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  const llvm::StringRef sourceText = twoOccurrences ? R"mlir(
    module {
      fabric.module @two_integer_adds(
          %a0: !fabric.bits<8>, %b0: !fabric.bits<8>,
          %a1: !fabric.bits<8>, %b1: !fabric.bits<8>)
          -> (!fabric.bits<8>, !fabric.bits<8>) {
        %pe0 = fabric.pe [spatial]
            (%pa0 = %a0 : !fabric.bits<8>, %pb0 = %b0 : !fabric.bits<8>)
            -> !fabric.bits<8> {
          %fu0 = fabric.fu
              (%fa0 = %pa0 : !fabric.bits<8>, %fb0 = %pb0 : !fabric.bits<8>)
              -> !fabric.bits<8> {
            %value0 = fabric.op [@arith.addi] (%fa0, %fb0)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [8 : i32]}}
              : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
            fabric.yield %value0 : !fabric.bits<8>
          }
        }
        %pe1 = fabric.pe [spatial]
            (%pa1 = %a1 : !fabric.bits<8>, %pb1 = %b1 : !fabric.bits<8>)
            -> !fabric.bits<8> {
          %fu1 = fabric.fu
              (%fa1 = %pa1 : !fabric.bits<8>, %fb1 = %pb1 : !fabric.bits<8>)
              -> !fabric.bits<8> {
            %value1 = fabric.op [@arith.addi] (%fa1, %fb1)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [8 : i32]}}
              : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
            fabric.yield %value1 : !fabric.bits<8>
          }
        }
        fabric.yield %pe0, %pe1 : !fabric.bits<8>, !fabric.bits<8>
      }
    }
  )mlir"
                                                    : R"mlir(
    module {
      fabric.module @integer_add(%a: !fabric.bits<8>, %b: !fabric.bits<8>)
          -> !fabric.bits<8> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<8>, %pb = %b : !fabric.bits<8>)
            -> !fabric.bits<8> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<8>, %fb = %pb : !fabric.bits<8>)
              -> !fabric.bits<8> {
            %value = fabric.op [@arith.addi] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [8 : i32]}}
              : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
            fabric.yield %value : !fabric.bits<8>
          }
        }
        fabric.yield %pe : !fabric.bits<8>
      }
    }
  )mlir";
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &context);
  require(test, static_cast<bool>(source),
          "unable to parse operation Fabric fixture");
  const std::vector<std::uint8_t> contract =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  const std::vector<std::int8_t> signedContract(contract.begin(),
                                                contract.end());
  source->walk([&](::fabric::OpOp operation) {
    operation->setAttr(::fabric::kResourceContractRecordAttrName,
                       mlir::DenseI8ArrayAttr::get(&context, signedContract));
  });
  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(test, static_cast<bool>(root),
          "operation Fabric fixture has no Module root");
  return take(test, loom::fabric::finalizeFabricRoot(root, store));
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

std::vector<FabricFuOccurrenceNodeRef>
findOperationOccurrences(llvm::StringRef test,
                         const loom::fabric::FabricArtifactView &view) {
  std::vector<FabricFuOccurrenceNodeRef> result;
  for (const auto occurrence : view.fuOccurrences()) {
    const auto definition = view.fuTemplateOf(occurrence);
    if (!definition)
      continue;
    const auto capabilities = view.resolvedFabricOpCapabilities(*definition);
    for (const auto &capability : capabilities)
      result.push_back(
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         view, capability.occurrence, occurrence)));
  }
  require(test, !result.empty(), "Fabric has no concrete operation occurrence");
  return result;
}

FinalizedConfigurationABI
makeEmptyConfigurationAbi(llvm::StringRef test, const ArtifactStore &store,
                          const FinalizedFabricRoot &fabric) {
  return take(test, loom::hardware::finalizeConfigurationABI(
                        {fabric.reference(), {}}, store));
}

void commonSkeletonRejectsUnresolvedOrUnboundLeaves() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  FinalizedFabricRoot fabric = makeOperationFabric(test, store);
  FinalizedConfigurationABI abi =
      makeEmptyConfigurationAbi(test, store, fabric);
  const std::vector<FabricFuOccurrenceNodeRef> occurrences =
      findOperationOccurrences(test, fabric.view());
  const FabricFuOccurrenceNodeRef occurrence = occurrences.front();

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
  std::vector<circt::hw::HWModuleGeneratedOp> leaves;
  std::vector<FabricOperationLeafAssociation> association;
  for (std::size_t index = 0; index < occurrences.size(); ++index) {
    const auto *capability =
        fabric.view().resolvedFabricOpCapability(occurrences[index]);
    require(test, capability != nullptr,
            "operation capability did not resolve");
    auto leaf = circt::hw::HWModuleGeneratedOp::create(
        builder, location,
        mlir::FlatSymbolRefAttr::get(
            &context,
            loom::hardware::rtl::fabricOperationGeneratorSchemaSymbol),
        builder.getStringAttr(
            (llvm::Twine("loom_fabric_operation_") + llvm::Twine(index)).str()),
        take(test, loom::hardware::rtl::deriveFabricOperationLeafPorts(
                       builder, *capability, abi.abi())));
    leaves.push_back(leaf);
    association.push_back({leaf, occurrences[index]});
  }
  circt::hw::HWModuleGeneratedOp leaf = leaves.front();
  const llvm::SmallVector<circt::hw::PortInfo> firstLeafPorts =
      leaf.getPortList();
  const std::vector<circt::hw::PortInfo> operationPorts(firstLeafPorts.begin(),
                                                        firstLeafPorts.end());
  circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("loom_common_skeleton_test"),
      circt::hw::ModulePortInfo({}, {}),
      [](mlir::OpBuilder &, circt::hw::HWModulePortAccessor &) {});

  if (llvm::Error error = loom::hardware::rtl::verifyCommonCirctSkeleton(
          *module, fabric.view(), abi.abi(), association))
    fail(test, llvm::toString(std::move(error)));

  const circt::hw::PortInfo unresolvedInput{
      {builder.getStringAttr("input"), builder.getI1Type(),
       circt::hw::ModulePort::Direction::Input}};
  const circt::hw::PortInfo unresolvedOutput{
      {builder.getStringAttr("output"), builder.getI1Type(),
       circt::hw::ModulePort::Direction::Output}};
  auto unresolvedTop = circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("unresolved_structural_top"),
      circt::hw::ModulePortInfo({unresolvedInput}, {unresolvedOutput}),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        llvm::SmallVector<mlir::Type> resultTypes{bodyBuilder.getI1Type()};
        llvm::SmallVector<mlir::Value> operands{accessor.getInput("input")};
        auto unresolved = mlir::UnrealizedConversionCastOp::create(
            bodyBuilder, location, resultTypes, operands);
        accessor.setOutput("output", unresolved.getResult(0));
      });
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.view(), abi.abi(), association),
              "unresolved structural lowering");
  expectError(test, loom::hardware::rtl::verifySpecializedCirctModule(*module),
              "unresolved structural lowering");
  expectError(
      test,
      loom::hardware::rtl::lowerAndExportSpecializedSystemVerilog(*module),
      "unresolved structural lowering");
  unresolvedTop.erase();

  const circt::hw::ModuleType exactLeafType = leaf.getModuleType();
  std::vector<circt::hw::ModulePort> wrongLeafPorts;
  wrongLeafPorts.reserve(operationPorts.size());
  for (const circt::hw::PortInfo &port : operationPorts)
    wrongLeafPorts.push_back(port);
  wrongLeafPorts.front().type = builder.getI1Type();
  leaf.setModuleType(circt::hw::ModuleType::get(&context, wrongLeafPorts));
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.view(), abi.abi(), association),
              "does not match its derived contract");
  leaf.setModuleType(exactLeafType);

  FinalizedFabricRoot foreignFabric = makeBoundaryOnlyFabric(test, store);
  FinalizedConfigurationABI foreignAbi =
      makeEmptyConfigurationAbi(test, store, foreignFabric);
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.view(), foreignAbi.abi(), association),
              "ConfigurationABI does not implement the exact Fabric");

  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.view(), abi.abi(), {}),
              "has no exact Fabric occurrence association");
  std::vector<FabricOperationLeafAssociation> duplicate = association;
  duplicate.push_back({leaf, occurrence});
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.view(), abi.abi(), duplicate),
              "associated more than once");

  auto secondLeaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(
          &context, loom::hardware::rtl::fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("loom_fabric_operation_1"), operationPorts);
  std::vector<FabricOperationLeafAssociation> duplicateOccurrence = association;
  duplicateOccurrence.push_back({secondLeaf, occurrence});
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.view(), abi.abi(), duplicateOccurrence),
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
      builder.getStringAttr("foreign_fabric_operation"), operationPorts);
  std::vector<FabricOperationLeafAssociation> foreignAssociation = association;
  foreignAssociation.front().module = foreignLeaf;
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.view(), abi.abi(), foreignAssociation),
              "does not name a Loom leaf in this module");

  schema.setDescriptor("unexpected.fabric.operation");
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.view(), abi.abi(), association),
              "schema has an unexpected descriptor");
  schema.setDescriptor(loom::hardware::rtl::fabricOperationGeneratorDescriptor);

  FabricFuOccurrenceNodeRef foreign = occurrence;
  foreign.ordinal += 1000000;
  std::vector<FabricOperationLeafAssociation> invalid = association;
  invalid.front().occurrence = foreign;
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *module, fabric.view(), abi.abi(), invalid),
              "does not resolve to a concrete Fabric operation capability");

  FinalizedFabricRoot twoOccurrenceFabric =
      makeOperationFabric(test, store, true);
  FinalizedConfigurationABI twoOccurrenceAbi =
      makeEmptyConfigurationAbi(test, store, twoOccurrenceFabric);
  const FabricFuOccurrenceNodeRef firstOfTwo =
      findOperationOccurrences(test, twoOccurrenceFabric.view()).front();
  const auto *firstOfTwoCapability =
      twoOccurrenceFabric.view().resolvedFabricOpCapability(firstOfTwo);
  require(test, firstOfTwoCapability != nullptr,
          "two-occurrence capability did not resolve");
  mlir::OwningOpRef<mlir::ModuleOp> incompleteModule =
      mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(incompleteModule->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location,
      loom::hardware::rtl::fabricOperationGeneratorSchemaSymbol,
      loom::hardware::rtl::fabricOperationGeneratorDescriptor,
      builder.getArrayAttr({}));
  auto incompleteLeaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(
          &context, loom::hardware::rtl::fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("incomplete_fabric_operation"),
      take(test, loom::hardware::rtl::deriveFabricOperationLeafPorts(
                     builder, *firstOfTwoCapability, twoOccurrenceAbi.abi())));
  expectError(test,
              loom::hardware::rtl::verifyCommonCirctSkeleton(
                  *incompleteModule, twoOccurrenceFabric.view(),
                  twoOccurrenceAbi.abi(), {{incompleteLeaf, firstOfTwo}}),
              "does not exactly cover Fabric operation occurrences");

  expectError(
      test,
      loom::hardware::rtl::lowerAndExportSpecializedSystemVerilog(*module),
      "unresolved Loom Fabric operation leaf");

  for (circt::hw::HWModuleGeneratedOp operationLeaf : leaves)
    operationLeaf.erase();
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
  FinalizedConfigurationABI abi =
      makeEmptyConfigurationAbi(test, store, fabric);

  mlir::MLIRContext firstContext;
  firstContext.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                           circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto first = take(test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                              firstContext, fabric.view(), abi.abi()));
  require(test, first.operationLeaves.empty(),
          "boundary-only skeleton invented an operation leaf");
  if (llvm::Error error = loom::hardware::rtl::verifyCommonCirctSkeleton(
          *first.module, fabric.view(), abi.abi(), first.operationLeaves))
    fail(test, llvm::toString(std::move(error)));

  mlir::MLIRContext secondContext;
  secondContext.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                            circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto second = take(test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                               secondContext, fabric.view(), abi.abi()));
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
              loom::hardware::rtl::buildModuleRootCirctSkeleton(
                  secondContext, system.view(), abi.abi()),
              "requires a Module root");
  FinalizedFabricRoot operationFabric = makeOperationFabric(test, store);
  FinalizedConfigurationABI operationAbi =
      makeEmptyConfigurationAbi(test, store, operationFabric);
  expectStructuralUnsupported(
      test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                secondContext, operationFabric.view(), operationAbi.abi()));
  expectError(test,
              loom::hardware::rtl::buildModuleRootCirctSkeleton(
                  secondContext, fabric.view(), operationAbi.abi()),
              "ConfigurationABI does not implement the exact Fabric");
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
