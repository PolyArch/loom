#include "Fabric/Identity/FabricPeConfiguration.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "Dataflow/IR/DataflowDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::fabric::FabricPeActive;
using loom::fabric::FabricPeConfigurationFieldKind;
using loom::fabric::FabricPeConfigurationValue;
using loom::fabric::FabricPeDisabled;
using loom::fabric::FabricPeDisconnected;
using loom::fabric::FabricPeInputDiscard;
using loom::fabric::FabricPeOutputDiscard;
using loom::fabric::FabricPeRoute;
using loom::fabric::FinalizedFabricRoot;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectRejected(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted an invalid PE configuration value");
  llvm::consumeError(value.takeError());
}

class TemporaryDirectory {
public:
  explicit TemporaryDirectory(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-fabric-pe-configuration-test", path))
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

mlir::MLIRContext &context() {
  static mlir::MLIRContext *result = [] {
    mlir::DialectRegistry registry;
    registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                    mlir::arith::ArithDialect>();
    auto *context =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    context->loadAllAvailableDialects();
    return context;
  }();
  return *result;
}

std::string moduleSource(bool reverse, bool includeMultiply = true) {
  const llvm::StringRef add = R"mlir(
    fabric.fu(%add_lhs = %pe_lhs : !fabric.bits<32>,
              %add_rhs = %pe_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
      %add_result = fabric.op [@arith.addi] (%add_lhs, %add_rhs)
        {implementation_family =
           #fabric.implementation_family<ScalarIntegerAddSub>,
         hw_params = {integer_widths = [32 : i32]}}
        : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %add_result : !fabric.bits<32>
    }
  )mlir";
  const llvm::StringRef multiply = R"mlir(
    fabric.fu(%mul_lhs = %pe_lhs : !fabric.bits<32>,
              %mul_rhs = %pe_rhs : !fabric.bits<32>) -> !fabric.bits<32> {
      %mul_result = fabric.op [@arith.muli] (%mul_lhs, %mul_rhs)
        {implementation_family =
           #fabric.implementation_family<ScalarIntegerMultiply>,
         hw_params = {integer_widths = [32 : i32]}}
        : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %mul_result : !fabric.bits<32>
    }
  )mlir";

  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << "module {\n"
         << "  fabric.module @root(%lhs: !fabric.bits<32>, "
            "%rhs: !fabric.bits<32>) {\n"
         << (includeMultiply ? "  %result:2 = fabric.pe [spatial]\n"
                             : "  %result = fabric.pe [spatial]\n")
         << "      (%pe_lhs = %lhs : !fabric.bits<32>,\n"
         << "       %pe_rhs = %rhs : !fabric.bits<32>)\n"
         << (includeMultiply
                 ? "      -> (!fabric.bits<32>, !fabric.bits<32>) {\n"
                 : "      -> (!fabric.bits<32>) {\n")
         << (includeMultiply && reverse ? multiply : add)
         << (includeMultiply ? (reverse ? add : multiply) : llvm::StringRef{})
         << "  }\n"
         << "  fabric.yield\n"
         << "  }\n"
         << "}\n";
  return stream.str();
}

FinalizedFabricRoot finalize(llvm::StringRef test, ArtifactStore &store,
                             bool reverse, bool includeMultiply = true) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          moduleSource(reverse, includeMultiply), &context());
  require(test, static_cast<bool>(module), "unable to parse PE fixture");
  const std::vector<std::uint8_t> contract =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  std::vector<std::int8_t> signedContract;
  signedContract.reserve(contract.size());
  for (std::uint8_t byte : contract)
    signedContract.push_back(static_cast<std::int8_t>(byte));
  module->walk([&](::fabric::OpOp operation) {
    operation->setAttr(::fabric::kResourceContractRecordAttrName,
                       mlir::DenseI8ArrayAttr::get(&context(), signedContract));
  });
  ::fabric::ModuleOp root;
  for (::fabric::ModuleOp candidate : module->getOps<::fabric::ModuleOp>())
    root = candidate;
  require(test, static_cast<bool>(root), "PE fixture has no Fabric root");
  return take(test, loom::fabric::finalizeFabricRoot(root, store));
}

template <typename Ref> bool canonicalLess(const Ref &lhs, const Ref &rhs) {
  return loom::fabric::canonicalFabricBytes(lhs) <
         loom::fabric::canonicalFabricBytes(rhs);
}

void staticSchemaIsCanonical() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  FinalizedFabricRoot forward = finalize(test, store, false);
  FinalizedFabricRoot reverse = finalize(test, store, true);
  FinalizedFabricRoot reduced = finalize(test, store, false, false);
  require(test, forward.view().identity() == reverse.view().identity(),
          "equivalent FU authoring order changed artifact identity");
  require(test, forward.view().identity() != reduced.view().identity(),
          "a different PE field relation preserved artifact identity");

  require(test, forward.view().peOccurrences().size() == 1,
          "fixture did not produce one PE occurrence");
  const auto pe = forward.view().peOccurrences().front();
  auto schema = take(test, forward.view().spatialPeConfigurationSchema(pe));
  require(test, schema.pe() == pe, "schema changed its PE owner");
  require(test, schema.fields().size() == 7,
          "schema did not contain activation and all FU port fields");
  auto reducedSchema =
      take(test, reduced.view().spatialPeConfigurationSchema(
                     reduced.view().peOccurrences().front()));
  require(test, reducedSchema.fields().size() == 4,
          "reduced fixture did not change the PE field relation");
  require(test,
          forward.view().inventorySize(
              loom::fabric::FabricInventoryOwnerRef::of(pe),
              loom::fabric::FabricInventoryKind::SemanticConfigField) == 7,
          "PE semantic-field inventory disagrees with its schema");

  const auto &fields = schema.fields();
  require(test,
          fields.front().kind == FabricPeConfigurationFieldKind::Activation &&
              !fields.front().port && fields.front().reference.ordinal == 0,
          "activation field is not canonical ordinal zero");
  for (std::size_t index = 1; index < 5; ++index)
    require(test,
            fields[index].kind ==
                    FabricPeConfigurationFieldKind::InputSelector &&
                fields[index].port &&
                fields[index].port->direction ==
                    loom::fabric::FabricPortDirection::Input,
            "input selector field has the wrong role");
  for (std::size_t index = 5; index < fields.size(); ++index)
    require(test,
            fields[index].kind ==
                    FabricPeConfigurationFieldKind::OutputSelector &&
                fields[index].port &&
                fields[index].port->direction ==
                    loom::fabric::FabricPortDirection::Output,
            "output selector field has the wrong role");
  for (std::size_t index = 1; index < 4; ++index)
    require(test, canonicalLess(*fields[index].port, *fields[index + 1].port),
            "input selector fields are not in canonical port order");
  require(test, canonicalLess(*fields[5].port, *fields[6].port),
          "output selector fields are not in canonical port order");
  for (std::size_t index = 0; index < fields.size(); ++index) {
    require(test, fields[index].reference.ordinal == index,
            "field reference ordinal is not dense");
    if (llvm::Error error = loom::fabric::validateFabricRef(
            forward.view(), fields[index].reference))
      fail(test, llvm::toString(std::move(error)));
  }

  auto reverseSchema = take(test, reverse.view().spatialPeConfigurationSchema(
                                      reverse.view().peOccurrences().front()));
  require(test, schema.fields() == reverseSchema.fields(),
          "authoring order changed the sealed PE schema");
}

void finiteDomainsAndCodecsAreExact() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  FinalizedFabricRoot finalized = finalize(test, store, false);
  const auto pe = finalized.view().peOccurrences().front();
  auto schema = take(test, finalized.view().spatialPeConfigurationSchema(pe));
  const auto &fields = schema.fields();

  auto activation = take(test, schema.finiteDomain(fields[0].reference));
  auto input = take(test, schema.finiteDomain(fields[1].reference));
  auto output = take(test, schema.finiteDomain(fields[5].reference));
  require(test,
          activation.size() == 3 && input.size() == 5 && output.size() == 4,
          "a PE field finite domain has the wrong cardinality");
  require(test, std::holds_alternative<FabricPeDisabled>(activation.front()),
          "activation domain does not begin with Disabled");
  require(test,
          std::holds_alternative<FabricPeDisconnected>(input.front()) &&
              std::holds_alternative<FabricPeDisconnected>(output.front()),
          "selector domain does not begin with Disconnected");
  require(test, std::holds_alternative<FabricPeOutputDiscard>(output.back()),
          "output selector domain does not end with payload-free Discard");

  for (const auto &field : fields) {
    auto domain = take(test, schema.finiteDomain(field.reference));
    for (const FabricPeConfigurationValue &value : domain) {
      auto encoded = take(test, schema.encode(field.reference, value));
      FabricPeConfigurationValue decoded =
          take(test, schema.decode(field.reference, encoded.bytes()));
      require(test, decoded == value, "PE field codec changed a domain value");
    }
  }

  const auto active = std::get<FabricPeActive>(activation[1]);
  const auto inputRoute = std::get<FabricPeRoute>(input[1]);
  const auto inputDiscard = std::get<FabricPeInputDiscard>(input[3]);
  const auto outputRoute = std::get<FabricPeRoute>(output[1]);
  const std::vector<std::uint8_t> tag0 = {0, 0, 0, 0};
  const std::vector<std::uint8_t> activeFu2 = {0, 0, 0, 1, 0, 0, 0, 3,
                                               0, 0, 0, 0, 0, 0, 0, 2};
  const std::vector<std::uint8_t> inputRoutePe5Endpoint0 = {
      0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0};
  const std::vector<std::uint8_t> inputDiscardPe5Endpoint0 = {
      0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0};
  const std::vector<std::uint8_t> outputRoutePe5Endpoint2 = {
      0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 2};
  const std::vector<std::uint8_t> tag2 = {0, 0, 0, 2};
  require(test,
          take(test, schema.encode(fields[0].reference, FabricPeDisabled{}))
              .bytes()
              .equals(tag0),
          "Disabled fixed byte vector changed");
  require(test,
          take(test, schema.encode(fields[0].reference, active))
              .bytes()
              .equals(activeFu2),
          "Active fixed byte vector changed");
  require(test,
          take(test, schema.encode(fields[1].reference, FabricPeDisconnected{}))
              .bytes()
              .equals(tag0),
          "Disconnected fixed byte vector changed");
  require(test,
          take(test, schema.encode(fields[1].reference, inputRoute))
              .bytes()
              .equals(inputRoutePe5Endpoint0),
          "Route fixed byte vector changed");
  require(test,
          take(test, schema.encode(fields[1].reference, inputDiscard))
              .bytes()
              .equals(inputDiscardPe5Endpoint0),
          "input Discard fixed byte vector changed");
  require(test,
          take(test, schema.encode(fields[5].reference, FabricPeDisconnected{}))
              .bytes()
              .equals(tag0),
          "output Disconnected fixed byte vector changed");
  require(test,
          take(test, schema.encode(fields[5].reference, outputRoute))
              .bytes()
              .equals(outputRoutePe5Endpoint2),
          "output Route fixed byte vector changed");
  require(
      test,
      take(test, schema.encode(fields[5].reference, FabricPeOutputDiscard{}))
          .bytes()
          .equals(tag2),
      "output Discard fixed byte vector changed");

  expectRejected(test, schema.decode(fields[0].reference, {0, 0, 0}));
  expectRejected(test, schema.decode(fields[0].reference, {0, 0, 0, 3}));
  expectRejected(test, schema.decode(fields[0].reference, {0, 0, 0, 0, 0}));
  expectRejected(
      test,
      schema.encode(fields[0].reference,
                    FabricPeActive{loom::fabric::FabricFuOccurrenceRef(999)}));
  expectRejected(
      test, schema.encode(fields[1].reference,
                          FabricPeRoute{schema.outputEndpoints().front()}));
  expectRejected(test, schema.encode(fields[1].reference,
                                     FabricPeInputDiscard{
                                         schema.outputEndpoints().front()}));
  expectRejected(test,
                 schema.encode(fields[5].reference,
                               FabricPeRoute{schema.inputEndpoints().front()}));
  expectRejected(
      test,
      schema.encode(fields[1].reference,
                    FabricPeRoute{loom::fabric::FabricTransportEndpointRef{
                        loom::fabric::FabricTransportEndpointOwnerRef::of(
                            loom::fabric::FabricPeOccurrenceRef(999)),
                        0}}));
  expectRejected(test,
                 schema.finiteDomain(loom::fabric::FabricSemanticConfigFieldRef{
                     loom::fabric::FabricConfigurationOwnerRef(
                         loom::fabric::FabricInventoryOwnerRef::of(pe)),
                     99}));
  expectRejected(test,
                 schema.finiteDomain(loom::fabric::FabricSemanticConfigFieldRef{
                     loom::fabric::FabricConfigurationOwnerRef(
                         loom::fabric::FabricInventoryOwnerRef::of(
                             loom::fabric::FabricPeOccurrenceRef(999))),
                     0}));
}

} // namespace

int main() {
  staticSchemaIsCanonical();
  finiteDomainsAndCodecsAreExact();
  return EXIT_SUCCESS;
}
