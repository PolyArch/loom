#include "Common/ArtifactStore.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Artifact/FabricClockResetValidation.h"
#include "Fabric/Artifact/FabricModuleRootView.h"
#include "Fabric/IR/FabricCanonicalEntity.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FuCapabilityDomain.h"
#include "Fabric/IR/MemoryConnectivityContract.h"
#include "Fabric/IR/ModuleDomain.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "FabricArtifactBytecodeInternal.h"
#include "FabricArtifactViewInternal.h"
#include "FabricModuleCanonicalPayload.h"
#include "FabricModuleViewBuilding.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectRejected(llvm::StringRef test, llvm::Expected<T> value,
                    llvm::StringRef diagnostic) {
  if (value)
    fail(test, "accepted a non-canonical Module domain carrier");
  const std::string message = llvm::toString(value.takeError());
  if (!llvm::StringRef(message).contains(diagnostic))
    fail(test, message);
}

class TemporaryDirectory final {
public:
  explicit TemporaryDirectory(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-fabric-domain-import-test", path))
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
  static mlir::MLIRContext *ctx = [] {
    mlir::DialectRegistry registry;
    registry.insert<::fabric::FabricDialect>();
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *ctx;
}

::fabric::ModuleOp root(llvm::StringRef test, mlir::ModuleOp module) {
  auto roots = module.getOps<::fabric::ModuleOp>();
  auto found = roots.begin();
  if (found == roots.end())
    fail(test, "fixture has no Fabric Module root");
  ::fabric::ModuleOp result = *found++;
  if (found != roots.end())
    fail(test, "fixture has more than one Fabric Module root");
  return result;
}

void nonCanonicalSlotPermutationIsRejected() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @domain(%left: !fabric.bits<8>,
                            %right: !fabric.bits<16>)
          -> (!fabric.bits<8>, !fabric.bits<16>) {
        fabric.yield %left, %right : !fabric.bits<8>, !fabric.bits<16>
      }
    }
  )mlir",
                                                        &context());
  if (!source)
    fail(test, "unable to parse the Module fixture");

  ::fabric::ModuleDomainAuthoringRelation relation;
  (void)take(test,
             relation.declareSlot(loom::fabric::FabricClockResetKind::Clock));
  (void)take(test,
             relation.declareSlot(loom::fabric::FabricClockResetKind::Clock));
  (void)take(test,
             relation.declareSlot(loom::fabric::FabricClockResetKind::Reset));
  for (loom::fabric::FabricPortDirection direction :
       {loom::fabric::FabricPortDirection::Input,
        loom::fabric::FabricPortDirection::Output}) {
    for (loom::fabric::FabricOrdinal ordinal = 0; ordinal != 2; ++ordinal) {
      if (llvm::Error error = relation.assignBoundary(
              direction, ordinal, loom::fabric::FabricClockResetKind::Clock,
              ordinal))
        fail(test, llvm::toString(std::move(error)));
      if (llvm::Error error = relation.assignBoundary(
              direction, ordinal, loom::fabric::FabricClockResetKind::Reset, 0))
        fail(test, llvm::toString(std::move(error)));
    }
  }

  loom::fabric::FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), relation,
                                                  store));
  loom::fabric::DecodedFabricArtifact decoded =
      take(test, loom::fabric::decodeFabricArtifactEnvelope(
                     finalized.canonicalBytes().bytes()));
  auto parsed = take(test, loom::fabric::detail::parseFabricBytecodeModule(
                               decoded.canonicalMlirBytecode));
  ::fabric::ModuleOp canonicalRoot = root(test, parsed.module.get());
  auto assignments = take(test, ::fabric::decodeModuleDomainAssignments(
                                    canonicalRoot.getDomainAssignmentsAttr()));
  for (loom::fabric::ModuleDomainAssignment &assignment : assignments)
    if (assignment.slot.kind == loom::fabric::FabricClockResetKind::Clock)
      assignment.slot.ordinal = 1 - assignment.slot.ordinal;
  llvm::sort(assignments, [](const auto &left, const auto &right) {
    return loom::fabric::canonicalFabricBytes(left) <
           loom::fabric::canonicalFabricBytes(right);
  });
  canonicalRoot.setDomainAssignmentsAttr(
      ::fabric::encodeModuleDomainAssignments(canonicalRoot.getContext(),
                                              assignments));

  decoded.canonicalMlirBytecode = take(
      test,
      loom::fabric::detail::writeCanonicalFabricBytecode(parsed.module.get()));
  loom::CanonicalSemanticBytes malformed =
      take(test, loom::fabric::encodeFabricArtifactEnvelope(
                     loom::fabric::FabricRootKind::Module, {},
                     decoded.canonicalMlirBytecode));
  const loom::ArtifactIdentity identity =
      take(test, store.put(loom::fabric::fabricArtifactSchema, malformed));
  expectRejected(test,
                 loom::fabric::importEntireFabricRoot(
                     {loom::fabric::fabricArtifactSchema.identity.str(),
                      loom::fabric::fabricArtifactSchema.version, identity},
                     store),
                 "canonical Module domain carrier is stale");
}

void nonCanonicalFuGraphOrderIsRejected() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @fu_domain(%data: !fabric.bits<32>) {
        %pe = fabric.pe [spatial]
            (%pe_data = %data : !fabric.bits<32>) -> !fabric.bits<32> {
          %fu = fabric.fu(%fu_data = %pe_data : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %first = fabric.mux %fu_data, %fu_data : !fabric.bits<32>
            %second = fabric.mux %fu_data, %fu_data : !fabric.bits<32>
            %sum = fabric.op [@arith.addi] (%first, %second)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [32 : i32]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %sum : !fabric.bits<32>
          }
        }
        fabric.yield
      }
    }
  )mlir",
                                                        &context());
  if (!source)
    fail(test, "unable to parse the multi-node FU fixture");

  const auto resourceBytes =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  std::vector<std::int8_t> signedResourceBytes;
  for (std::uint8_t byte : resourceBytes)
    signedResourceBytes.push_back(static_cast<std::int8_t>(byte));
  source->walk([&](::fabric::OpOp operation) {
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        mlir::DenseI8ArrayAttr::get(source->getContext(), signedResourceBytes));
  });
  ::fabric::FuOp fu;
  source->walk([&](::fabric::FuOp candidate) { fu = candidate; });
  auto capabilityDomain = take(test, ::fabric::FuCapabilityDomainRecord::create(
                                         {{{2}, {{0, 0}, {1, 0}}}}));
  const auto capabilityBytes =
      take(test, ::fabric::encodeFuCapabilityDomainRecord(capabilityDomain));
  std::vector<std::int8_t> signedCapabilityBytes;
  for (std::uint8_t byte : capabilityBytes)
    signedCapabilityBytes.push_back(static_cast<std::int8_t>(byte));
  fu.setCapabilityTemplatesAttr(::fabric::FuCapabilityDomainAttr::get(
      source->getContext(), mlir::DenseI8ArrayAttr::get(
                                source->getContext(), signedCapabilityBytes)));

  loom::fabric::FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), store));
  loom::fabric::DecodedFabricArtifact decoded =
      take(test, loom::fabric::decodeFabricArtifactEnvelope(
                     finalized.canonicalBytes().bytes()));
  auto parsed = take(test, loom::fabric::detail::parseFabricBytecodeModule(
                               decoded.canonicalMlirBytecode));
  llvm::SmallVector<mlir::Operation *, 2> muxes;
  root(test, parsed.module.get())->walk([&](::fabric::MuxOp mux) {
    muxes.push_back(mux.getOperation());
  });
  if (muxes.size() != 2)
    fail(test, "canonical multi-node FU has the wrong mux inventory");
  muxes[1]->moveBefore(muxes[0]);

  decoded.canonicalMlirBytecode = take(
      test,
      loom::fabric::detail::writeCanonicalFabricBytecode(parsed.module.get()));
  loom::CanonicalSemanticBytes malformed =
      take(test, loom::fabric::encodeFabricArtifactEnvelope(
                     loom::fabric::FabricRootKind::Module, {},
                     decoded.canonicalMlirBytecode));
  const loom::ArtifactIdentity identity =
      take(test, store.put(loom::fabric::fabricArtifactSchema, malformed));
  expectRejected(test,
                 loom::fabric::importEntireFabricRoot(
                     {loom::fabric::fabricArtifactSchema.identity.str(),
                      loom::fabric::fabricArtifactSchema.version, identity},
                     store),
                 "canonical Module graph operation order is not canonical");
}

void canonicalFuDefinitionOrdinalsSurviveStrictImport() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @fu_definition_ordinal(
          %first_data: !fabric.bits<32>, %second_data: !fabric.bits<32>) {
        %first_pe = fabric.pe [spatial]
            (%pe_data = %first_data : !fabric.bits<32>) -> !fabric.bits<32> {
          %first_fu = fabric.fu(%fu_data = %pe_data : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %left = fabric.mux %fu_data, %fu_data : !fabric.bits<32>
            %right = fabric.mux %fu_data, %fu_data : !fabric.bits<32>
            %product = fabric.op [@arith.muli] (%fu_data, %fu_data)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerMultiply>,
               hw_params = {integer_widths = [32 : i32]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %product : !fabric.bits<32>
          }
        }
        %second_pe = fabric.pe [spatial]
            (%pe_data = %second_data : !fabric.bits<32>) -> !fabric.bits<32> {
          %second_fu = fabric.fu(%fu_data = %pe_data : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %left = fabric.mux %fu_data, %fu_data : !fabric.bits<32>
            %right = fabric.mux %fu_data, %fu_data : !fabric.bits<32>
            %product = fabric.op [@arith.muli] (%fu_data, %fu_data)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerMultiply>,
               hw_params = {integer_widths = [32 : i32]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %product : !fabric.bits<32>
          }
        }
        fabric.yield
      }
    }
  )mlir",
                                                        &context());
  if (!source)
    fail(test, "unable to parse the FU definition-ordinal fixture");

  llvm::SmallVector<::fabric::PeOp, 2> pes;
  llvm::SmallVector<::fabric::FuOp, 2> fus;
  source->walk([&](::fabric::PeOp pe) { pes.push_back(pe); });
  source->walk([&](::fabric::FuOp fu) { fus.push_back(fu); });
  if (pes.size() != 2 || fus.size() != 2)
    fail(test, "FU definition-ordinal fixture has the wrong inventory");
  llvm::SmallVector<llvm::SmallVector<mlir::Operation *, 3>, 2>
      nodesByOccurrence;
  for (::fabric::FuOp fu : fus) {
    llvm::SmallVector<mlir::Operation *, 3> nodes;
    fu->walk([&](mlir::Operation *operation) {
      if (mlir::isa<::fabric::OpOp, ::fabric::MuxOp, ::fabric::DemuxOp>(
              operation))
        nodes.push_back(operation);
    });
    if (nodes.size() != 3)
      fail(test, "FU definition-ordinal fixture has the wrong node inventory");
    nodesByOccurrence.push_back(std::move(nodes));
  }

  const auto resourceBytes =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  std::vector<std::int8_t> signedResourceBytes;
  signedResourceBytes.reserve(resourceBytes.size());
  for (std::uint8_t byte : resourceBytes)
    signedResourceBytes.push_back(static_cast<std::int8_t>(byte));
  source->walk([&](::fabric::OpOp operation) {
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        mlir::DenseI8ArrayAttr::get(source->getContext(), signedResourceBytes));
  });

  auto capability =
      take(test, ::fabric::FuCapabilityDomainRecord::create({{{2}, {}}}));
  const auto capabilityBytes =
      take(test, ::fabric::encodeFuCapabilityDomainRecord(capability));
  std::vector<std::int8_t> signedCapabilityBytes;
  signedCapabilityBytes.reserve(capabilityBytes.size());
  for (std::uint8_t byte : capabilityBytes)
    signedCapabilityBytes.push_back(static_cast<std::int8_t>(byte));
  for (::fabric::FuOp fu : fus)
    fu.setCapabilityTemplatesAttr(::fabric::FuCapabilityDomainAttr::get(
        source->getContext(),
        mlir::DenseI8ArrayAttr::get(source->getContext(),
                                    signedCapabilityBytes)));

  using Kind = loom::fabric::FabricClockResetKind;
  using Role = ::fabric::ModuleDomainAuthoringRelation::InternalMemberRole;
  ::fabric::ModuleDomainAuthoringRelation relation;
  const auto firstClock = take(test, relation.declareSlot(Kind::Clock));
  const auto secondClock = take(test, relation.declareSlot(Kind::Clock));
  const auto firstReset = take(test, relation.declareSlot(Kind::Reset));
  const auto secondReset = take(test, relation.declareSlot(Kind::Reset));
  const auto assign = [&](mlir::Operation *owner, Role role,
                          loom::fabric::FabricOrdinal memberOrdinal,
                          bool secondDomain) {
    if (llvm::Error error =
            relation.noteInternalMember(owner, role, memberOrdinal))
      fail(test, llvm::toString(std::move(error)));
    for (Kind kind : {Kind::Clock, Kind::Reset}) {
      const loom::fabric::FabricOrdinal slot =
          kind == Kind::Clock ? (secondDomain ? secondClock : firstClock)
                              : (secondDomain ? secondReset : firstReset);
      if (llvm::Error error =
              relation.assignInternal(owner, role, memberOrdinal, kind, slot))
        fail(test, llvm::toString(std::move(error)));
    }
  };
  for (loom::fabric::FabricOrdinal ordinal = 0; ordinal != 2; ++ordinal)
    for (Kind kind : {Kind::Clock, Kind::Reset})
      if (llvm::Error error = relation.assignBoundary(
              loom::fabric::FabricPortDirection::Input, ordinal, kind,
              kind == Kind::Clock ? (ordinal == 0 ? firstClock : secondClock)
                                  : (ordinal == 0 ? firstReset : secondReset)))
        fail(test, llvm::toString(std::move(error)));
  for (auto [occurrence, fu] : llvm::enumerate(fus)) {
    const bool occurrenceSecond = occurrence != 0;
    assign(pes[occurrence], Role::Occurrence, 0, occurrenceSecond);
    assign(pes[occurrence], Role::InstructionContext, 0, occurrenceSecond);
    assign(fu, Role::Occurrence, 0, occurrenceSecond);
    for (auto [nodeOrdinal, node] :
         llvm::enumerate(nodesByOccurrence[occurrence]))
      assign(node, Role::FuNode, 0, (occurrence + nodeOrdinal) % 2 != 0);
  }

  loom::fabric::FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), relation,
                                                  store));
  if (finalized.view().fuTemplates().size() != 1 ||
      finalized.view().fuOccurrences().size() != 2)
    fail(test, "finalized FU definition-ordinal inventory is incomplete");
  const auto finalizedModule =
      take(test, loom::fabric::requireModuleRoot(finalized.view()));
  const loom::fabric::FabricFuTemplateRef definition =
      finalized.view().fuTemplates().front();
  const auto definitionOwner =
      loom::fabric::FabricInventoryOwnerRef::of(definition);
  const std::uint64_t nodeCount = finalized.view().inventorySize(
      definitionOwner, loom::fabric::FabricInventoryKind::FuNode);
  if (nodeCount != 3)
    fail(test, "finalized FU definition has the wrong node inventory");
  std::vector<std::vector<
      std::pair<loom::fabric::FabricOrdinal, loom::fabric::FabricOrdinal>>>
      occurrenceDomains;
  for (loom::fabric::FabricFuOccurrenceRef occurrence :
       finalized.view().fuOccurrences()) {
    std::vector<
        std::pair<loom::fabric::FabricOrdinal, loom::fabric::FabricOrdinal>>
        domains;
    for (loom::fabric::FabricOrdinal ordinal = 0; ordinal != nodeCount;
         ++ordinal) {
      const auto kind = finalized.view().fuNodeKind(definitionOwner, ordinal);
      if (!kind)
        fail(test, "finalized FU definition has an untyped node");
      const auto occurrenceNode =
          take(test,
               loom::fabric::deriveFabricFuOccurrenceNode(
                   finalized.view(), {*kind, definition, ordinal}, occurrence));
      const auto physical = take(
          test,
          loom::fabric::FabricModulePhysicalOwnerRef::create(occurrenceNode));
      const auto member =
          loom::fabric::FabricModuleDomainMemberRef::of(physical);
      std::optional<loom::fabric::FabricOrdinal> clock;
      std::optional<loom::fabric::FabricOrdinal> reset;
      for (const loom::fabric::ModuleDomainAssignment &assignment :
           finalizedModule.domainAssignments()) {
        if (assignment.member != member)
          continue;
        if (assignment.slot.kind == Kind::Clock)
          clock = assignment.slot.ordinal;
        else if (assignment.slot.kind == Kind::Reset)
          reset = assignment.slot.ordinal;
      }
      if (!clock || !reset)
        fail(test, "finalized FU occurrence node has no complete domain");
      domains.emplace_back(*clock, *reset);
    }
    occurrenceDomains.push_back(std::move(domains));
  }
  for (loom::fabric::FabricOrdinal ordinal = 0; ordinal != nodeCount; ++ordinal)
    if (occurrenceDomains[0][ordinal].first ==
            occurrenceDomains[1][ordinal].first ||
        occurrenceDomains[0][ordinal].second ==
            occurrenceDomains[1][ordinal].second)
      fail(test, "definition node changed occurrence-domain correspondence");
  const auto templates = finalized.view().fuCapabilityTemplates(definition);
  if (templates.size() != 1 || templates.front().activeNodes.size() != 1)
    fail(test, "finalized FU capability changed its active-node domain");
  const loom::fabric::FabricFuTemplateNodeRef templateNode =
      templates.front().activeNodes.front();
  if (templateNode.node != loom::fabric::FabricFuNodeKind::Op)
    fail(test, "finalized FU capability selected a non-operation node");
  for (loom::fabric::FabricFuOccurrenceRef occurrence :
       finalized.view().fuOccurrences()) {
    const auto occurrenceNode =
        take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                       finalized.view(), templateNode, occurrence));
    if (!finalized.view().resolvedFabricOpCapability(occurrenceNode) ||
        !finalized.view().resourceContract(
            loom::fabric::FabricInventoryOwnerRef::of(occurrenceNode)))
      fail(test, "occurrence node lost its definition-owned contracts");
  }

  loom::fabric::FinalizedFabricRoot imported = take(
      test, loom::fabric::importEntireFabricRoot(finalized.reference(), store));
  const auto importedModule =
      take(test, loom::fabric::requireModuleRoot(imported.view()));
  if (imported.view().fuCapabilityTemplates(definition) != templates ||
      importedModule.domainAssignments() != finalizedModule.domainAssignments())
    fail(test, "strict import changed FU definition-local projections");
}

void fabricOpSchemaInventoryIsCanonicalAndStrict() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @op_list(%left: !fabric.bits<32>,
                             %right: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pe_left = %left : !fabric.bits<32>,
             %pe_right = %right : !fabric.bits<32>) -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fu_left = %pe_left : !fabric.bits<32>,
               %fu_right = %pe_right : !fabric.bits<32>) -> !fabric.bits<32> {
            %sum = fabric.op [@arith.subi, @arith.addi] (%fu_left, %fu_right)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [32 : i32]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %sum : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir",
                                                        &context());
  if (!source)
    fail(test, "unable to parse the operation-schema fixture");

  const auto resourceBytes =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  std::vector<std::int8_t> signedResourceBytes;
  for (std::uint8_t byte : resourceBytes)
    signedResourceBytes.push_back(static_cast<std::int8_t>(byte));
  source->walk([&](::fabric::OpOp operation) {
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        mlir::DenseI8ArrayAttr::get(source->getContext(), signedResourceBytes));
  });

  loom::fabric::FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), store));
  loom::fabric::DecodedFabricArtifact decoded =
      take(test, loom::fabric::decodeFabricArtifactEnvelope(
                     finalized.canonicalBytes().bytes()));
  auto parsed = take(test, loom::fabric::detail::parseFabricBytecodeModule(
                               decoded.canonicalMlirBytecode));
  ::fabric::OpOp operation;
  root(test, parsed.module.get())->walk([&](::fabric::OpOp found) {
    operation = found;
  });
  if (!operation || operation.getOpList().size() != 2)
    fail(test, "canonical operation-schema inventory is incomplete");

  std::optional<std::vector<std::uint8_t>> previous;
  for (mlir::Attribute attribute : operation.getOpList()) {
    auto symbol = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(attribute);
    if (!symbol)
      fail(test, "canonical operation-schema inventory is not symbolic");
    auto schema = dataflow::findOperationSchema(symbol.getValue());
    if (!schema)
      fail(test, "canonical operation-schema inventory is not registered");
    std::vector<std::uint8_t> identity =
        take(test, dataflow::encodeOperationSchemaId(*schema)).bytes().vec();
    if (previous && !(*previous < identity))
      fail(test, "finalizer did not canonicalize op_list by schema identity");
    previous = std::move(identity);
  }

  llvm::SmallVector<mlir::Attribute, 2> reversed(operation.getOpList().begin(),
                                                 operation.getOpList().end());
  std::reverse(reversed.begin(), reversed.end());
  operation->setAttr("op_list",
                     mlir::ArrayAttr::get(operation.getContext(), reversed));
  decoded.canonicalMlirBytecode = take(
      test,
      loom::fabric::detail::writeCanonicalFabricBytecode(parsed.module.get()));
  loom::CanonicalSemanticBytes malformed =
      take(test, loom::fabric::encodeFabricArtifactEnvelope(
                     loom::fabric::FabricRootKind::Module, {},
                     decoded.canonicalMlirBytecode));
  const loom::ArtifactIdentity identity =
      take(test, store.put(loom::fabric::fabricArtifactSchema, malformed));
  expectRejected(test,
                 loom::fabric::importEntireFabricRoot(
                     {loom::fabric::fabricArtifactSchema.identity.str(),
                      loom::fabric::fabricArtifactSchema.version, identity},
                     store),
                 "fabric.op op_list is not in canonical schema-ID order");
}

void missingCanonicalFuCapabilityCarrierIsRejected() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @fu_capability(%data: !fabric.bits<32>) {
        %pe = fabric.pe [spatial]
            (%pe_data = %data : !fabric.bits<32>) -> !fabric.bits<32> {
          %fu = fabric.fu(%fu_data = %pe_data : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %sum = fabric.op [@arith.addi] (%fu_data, %fu_data)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [32 : i32]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %sum : !fabric.bits<32>
          }
        }
        fabric.yield
      }
    }
  )mlir",
                                                        &context());
  if (!source)
    fail(test, "unable to parse the FU capability fixture");

  const auto resourceBytes =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  std::vector<std::int8_t> signedResourceBytes;
  for (std::uint8_t byte : resourceBytes)
    signedResourceBytes.push_back(static_cast<std::int8_t>(byte));
  source->walk([&](::fabric::OpOp operation) {
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        mlir::DenseI8ArrayAttr::get(source->getContext(), signedResourceBytes));
  });

  loom::fabric::FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), store));
  loom::fabric::DecodedFabricArtifact decoded =
      take(test, loom::fabric::decodeFabricArtifactEnvelope(
                     finalized.canonicalBytes().bytes()));
  auto parsed = take(test, loom::fabric::detail::parseFabricBytecodeModule(
                               decoded.canonicalMlirBytecode));
  ::fabric::FuOp fu;
  root(test, parsed.module.get())->walk(
      [&](::fabric::FuOp found) { fu = found; });
  if (!fu || !fu.getCapabilityTemplatesAttr())
    fail(test, "finalizer did not materialize the FU capability carrier");
  fu.removeCapabilityTemplatesAttr();

  decoded.canonicalMlirBytecode = take(
      test,
      loom::fabric::detail::writeCanonicalFabricBytecode(parsed.module.get()));
  loom::CanonicalSemanticBytes malformed =
      take(test, loom::fabric::encodeFabricArtifactEnvelope(
                     loom::fabric::FabricRootKind::Module, {},
                     decoded.canonicalMlirBytecode));
  const loom::ArtifactIdentity identity =
      take(test, store.put(loom::fabric::fabricArtifactSchema, malformed));
  expectRejected(test,
                 loom::fabric::importEntireFabricRoot(
                     {loom::fabric::fabricArtifactSchema.identity.str(),
                      loom::fabric::fabricArtifactSchema.version, identity},
                     store),
                 "canonical FU capability domain carrier is stale");
}

void nonCanonicalModuleGraphOrderIsRejected() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @graph_order(%left: !fabric.bits<32>,
                                 %right: !fabric.bits<32>)
          -> (!fabric.bits<32>, !fabric.bits<32>) {
        %first = fabric.fifo %left [max_depth = 2, bypassable = true]
            : !fabric.bits<32>
        %second = fabric.fifo %right [max_depth = 3, bypassable = false]
            : !fabric.bits<32>
        fabric.yield %first, %second : !fabric.bits<32>, !fabric.bits<32>
      }
    }
  )mlir",
                                                        &context());
  if (!source)
    fail(test, "unable to parse the graph-order fixture");

  loom::fabric::FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), store));
  loom::fabric::DecodedFabricArtifact decoded =
      take(test, loom::fabric::decodeFabricArtifactEnvelope(
                     finalized.canonicalBytes().bytes()));
  auto parsed = take(test, loom::fabric::detail::parseFabricBytecodeModule(
                               decoded.canonicalMlirBytecode));
  llvm::SmallVector<::fabric::FifoOp, 2> fifos;
  root(test, parsed.module.get())->walk([&](::fabric::FifoOp fifo) {
    fifos.push_back(fifo);
  });
  if (fifos.size() != 2)
    fail(test, "canonical graph-order fixture has the wrong FIFO inventory");
  fifos[1]->moveBefore(fifos[0]);

  decoded.canonicalMlirBytecode = take(
      test,
      loom::fabric::detail::writeCanonicalFabricBytecode(parsed.module.get()));
  loom::CanonicalSemanticBytes malformed =
      take(test, loom::fabric::encodeFabricArtifactEnvelope(
                     loom::fabric::FabricRootKind::Module, {},
                     decoded.canonicalMlirBytecode));
  const loom::ArtifactIdentity identity =
      take(test, store.put(loom::fabric::fabricArtifactSchema, malformed));
  expectRejected(test,
                 loom::fabric::importEntireFabricRoot(
                     {loom::fabric::fabricArtifactSchema.identity.str(),
                      loom::fabric::fabricArtifactSchema.version, identity},
                     store),
                 "canonical Module graph operation order is not canonical");
}

void uncoordinatedStoredEntityIdPermutationIsRejected() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @entity_ids(%left: !fabric.bits<8>,
                                %right: !fabric.bits<8>)
          -> (!fabric.bits<8>, !fabric.bits<8>) {
        %left_fifo = fabric.fifo %left [max_depth = 2, bypassable = false]
            : !fabric.bits<8>
        %right_fifo = fabric.fifo %right [max_depth = 2, bypassable = false]
            : !fabric.bits<8>
        fabric.yield %left_fifo, %right_fifo
            : !fabric.bits<8>, !fabric.bits<8>
      }
    }
  )mlir",
                                                        &context());
  if (!source)
    fail(test, "unable to parse the entity-ID fixture");

  ::fabric::ModuleDomainAuthoringRelation relation;
  using Kind = loom::fabric::FabricClockResetKind;
  using Direction = loom::fabric::FabricPortDirection;
  using Role = ::fabric::ModuleDomainAuthoringRelation::InternalMemberRole;
  (void)take(test, relation.declareSlot(Kind::Clock));
  (void)take(test, relation.declareSlot(Kind::Clock));
  (void)take(test, relation.declareSlot(Kind::Reset));
  for (Direction direction : {Direction::Input, Direction::Output})
    for (loom::fabric::FabricOrdinal ordinal = 0; ordinal != 2; ++ordinal) {
      if (llvm::Error error =
              relation.assignBoundary(direction, ordinal, Kind::Clock, ordinal))
        fail(test, llvm::toString(std::move(error)));
      if (llvm::Error error =
              relation.assignBoundary(direction, ordinal, Kind::Reset, 0))
        fail(test, llvm::toString(std::move(error)));
    }
  llvm::SmallVector<::fabric::FifoOp, 2> draftFifos;
  source->walk(
      [&](::fabric::FifoOp fifo) { draftFifos.push_back(fifo); });
  if (draftFifos.size() != 2)
    fail(test, "entity-ID fixture has the wrong FIFO inventory");
  for (auto [ordinal, fifo] : llvm::enumerate(draftFifos)) {
    if (llvm::Error error =
            relation.noteInternalMember(fifo, Role::Occurrence, 0))
      fail(test, llvm::toString(std::move(error)));
    if (llvm::Error error = relation.assignInternal(
            fifo, Role::Occurrence, 0, Kind::Clock, ordinal))
      fail(test, llvm::toString(std::move(error)));
    if (llvm::Error error =
            relation.assignInternal(fifo, Role::Occurrence, 0, Kind::Reset, 0))
      fail(test, llvm::toString(std::move(error)));
  }

  loom::fabric::FinalizedFabricRoot finalized = take(
      test, loom::fabric::finalizeFabricRoot(root(test, *source), relation,
                                             store));
  loom::fabric::DecodedFabricArtifact decoded =
      take(test, loom::fabric::decodeFabricArtifactEnvelope(
                     finalized.canonicalBytes().bytes()));
  auto parsed = take(test, loom::fabric::detail::parseFabricBytecodeModule(
                               decoded.canonicalMlirBytecode));
  llvm::SmallVector<::fabric::FifoOp, 2> canonicalFifos;
  root(test, parsed.module.get())->walk(
      [&](::fabric::FifoOp fifo) { canonicalFifos.push_back(fifo); });
  if (canonicalFifos.size() != 2)
    fail(test, "canonical entity-ID fixture has the wrong FIFO inventory");
  mlir::Attribute firstId =
      canonicalFifos[0]->getAttr(::fabric::kEntityIdAttrName);
  mlir::Attribute secondId =
      canonicalFifos[1]->getAttr(::fabric::kEntityIdAttrName);
  if (!firstId || !secondId)
    fail(test, "canonical entity-ID fixture has no entity IDs");
  canonicalFifos[0]->setAttr(::fabric::kEntityIdAttrName, secondId);
  canonicalFifos[1]->setAttr(::fabric::kEntityIdAttrName, firstId);

  decoded.canonicalMlirBytecode = take(
      test,
      loom::fabric::detail::writeCanonicalFabricBytecode(parsed.module.get()));
  loom::CanonicalSemanticBytes malformed =
      take(test, loom::fabric::encodeFabricArtifactEnvelope(
                     loom::fabric::FabricRootKind::Module, {},
                     decoded.canonicalMlirBytecode));
  const loom::ArtifactIdentity identity =
      take(test, store.put(loom::fabric::fabricArtifactSchema, malformed));
  expectRejected(test,
                 loom::fabric::importEntireFabricRoot(
                     {loom::fabric::fabricArtifactSchema.identity.str(),
                      loom::fabric::fabricArtifactSchema.version, identity},
                     store),
                 "canonical Module domain carrier is stale");
}

void derivedIdentifiersOnNonCarriersAreRejected() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @identity(%value: !fabric.bits<8>) -> !fabric.bits<8> {
        fabric.yield %value : !fabric.bits<8>
      }
    }
  )mlir",
                                                        &context());
  if (!source)
    fail(test, "unable to parse the derived-identifier fixture");
  loom::fabric::FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), store));
  const llvm::StringLiteral attributes[] = {
      ::fabric::kEntityIdAttrName, ::fabric::kFuTemplateIdAttrName,
      ::fabric::kMemoryEngineTemplateIdAttrName};
  for (llvm::StringLiteral attribute : attributes) {
    loom::fabric::DecodedFabricArtifact decoded =
        take(test, loom::fabric::decodeFabricArtifactEnvelope(
                       finalized.canonicalBytes().bytes()));
    auto parsed = take(test, loom::fabric::detail::parseFabricBytecodeModule(
                                 decoded.canonicalMlirBytecode));
    ::fabric::ModuleOp canonicalRoot = root(test, parsed.module.get());
    canonicalRoot.getBody().front().getTerminator()->setAttr(
        attribute,
        ::fabric::EntityIdAttr::get(canonicalRoot.getContext(), 123));
    decoded.canonicalMlirBytecode =
        take(test, loom::fabric::detail::writeCanonicalFabricBytecode(
                       parsed.module.get()));
    loom::CanonicalSemanticBytes malformed =
        take(test, loom::fabric::encodeFabricArtifactEnvelope(
                       loom::fabric::FabricRootKind::Module, {},
                       decoded.canonicalMlirBytecode));
    const loom::ArtifactIdentity identity =
        take(test, store.put(loom::fabric::fabricArtifactSchema, malformed));
    expectRejected(test,
                   loom::fabric::importEntireFabricRoot(
                       {loom::fabric::fabricArtifactSchema.identity.str(),
                        loom::fabric::fabricArtifactSchema.version, identity},
                       store),
                   "derived identifier is attached to a non-carrier");
  }
}

void canonicalRelationsOnNonCarriersAreRejected() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @relations(%value: !fabric.bits<8>) -> !fabric.bits<8> {
        fabric.yield %value : !fabric.bits<8>
      }
    }
  )mlir",
                                                        &context());
  if (!source)
    fail(test, "unable to parse the canonical-relation fixture");
  loom::fabric::FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), store));
  loom::fabric::DecodedFabricArtifact decoded =
      take(test, loom::fabric::decodeFabricArtifactEnvelope(
                     finalized.canonicalBytes().bytes()));
  auto parsed = take(test, loom::fabric::detail::parseFabricBytecodeModule(
                               decoded.canonicalMlirBytecode));
  ::fabric::ModuleOp canonicalRoot = root(test, parsed.module.get());
  mlir::Operation *nonCarrier =
      canonicalRoot.getBody().front().getTerminator();

  auto capability = take(
      test, ::fabric::FuCapabilityDomainRecord::create({{{0}, {}}}));
  const std::vector<std::uint8_t> capabilityBytes =
      take(test, ::fabric::encodeFuCapabilityDomainRecord(capability));
  const std::vector<std::int8_t> signedCapabilityBytes(capabilityBytes.begin(),
                                                       capabilityBytes.end());
  const std::pair<llvm::StringRef, mlir::Attribute> relations[] = {
      {"domain_slots", canonicalRoot.getDomainSlotsAttr()},
      {"domain_assignments", canonicalRoot.getDomainAssignmentsAttr()},
      {"capability_templates",
       ::fabric::FuCapabilityDomainAttr::get(
           canonicalRoot.getContext(),
           mlir::DenseI8ArrayAttr::get(canonicalRoot.getContext(),
                                       signedCapabilityBytes))}};

  for (const auto &[name, relation] : relations) {
    nonCarrier->setAttr(name, relation);
    decoded.canonicalMlirBytecode = take(
        test, loom::fabric::detail::writeCanonicalFabricBytecode(
                  parsed.module.get()));
    loom::CanonicalSemanticBytes malformed =
        take(test, loom::fabric::encodeFabricArtifactEnvelope(
                       loom::fabric::FabricRootKind::Module, {},
                       decoded.canonicalMlirBytecode));
    const loom::ArtifactIdentity identity =
        take(test, store.put(loom::fabric::fabricArtifactSchema, malformed));
    expectRejected(test,
                   loom::fabric::importEntireFabricRoot(
                       {loom::fabric::fabricArtifactSchema.identity.str(),
                        loom::fabric::fabricArtifactSchema.version, identity},
                       store),
                   "canonical relation is attached to a non-carrier");
    nonCarrier->removeAttr(name);
  }
}

void authoringStateAndDeclarationsAreRejected() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @authoring(%value: !fabric.bits<8>) -> !fabric.bits<8> {
        %fifo = fabric.fifo %value [max_depth = 2, bypassable = true]
            : !fabric.bits<8>
        fabric.yield %fifo : !fabric.bits<8>
      }
    }
  )mlir",
                                                        &context());
  if (!source)
    fail(test, "unable to parse the authoring-state fixture");
  loom::fabric::FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), store));

  const auto rejectMutation = [&](llvm::StringRef diagnostic, auto mutate) {
    loom::fabric::DecodedFabricArtifact decoded =
        take(test, loom::fabric::decodeFabricArtifactEnvelope(
                       finalized.canonicalBytes().bytes()));
    auto parsed = take(test, loom::fabric::detail::parseFabricBytecodeModule(
                                 decoded.canonicalMlirBytecode));
    mutate(root(test, parsed.module.get()));
    decoded.canonicalMlirBytecode =
        take(test, loom::fabric::detail::writeCanonicalFabricBytecode(
                       parsed.module.get()));
    loom::CanonicalSemanticBytes malformed =
        take(test, loom::fabric::encodeFabricArtifactEnvelope(
                       loom::fabric::FabricRootKind::Module, {},
                       decoded.canonicalMlirBytecode));
    const loom::ArtifactIdentity identity =
        take(test, store.put(loom::fabric::fabricArtifactSchema, malformed));
    expectRejected(test,
                   loom::fabric::importEntireFabricRoot(
                       {loom::fabric::fabricArtifactSchema.identity.str(),
                        loom::fabric::fabricArtifactSchema.version, identity},
                       store),
                   diagnostic);
  };

  rejectMutation(
      "canonical Module payload retains authoring-only state",
      [&](::fabric::ModuleOp canonicalRoot) {
        ::fabric::FifoOp fifo;
        canonicalRoot->walk([&](::fabric::FifoOp found) { fifo = found; });
        if (!fifo)
          fail(test, "canonical authoring-state fixture has no FIFO");
        fifo->setAttr("bypassed",
                      mlir::BoolAttr::get(canonicalRoot.getContext(), true));
      });

  auto declarationSource = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.switch @unused [spatial]
          (!fabric.bits<8>) -> (!fabric.bits<8>)
          [{connectivity_table = ["1"]}]
    }
  )mlir",
                                                                   &context());
  if (!declarationSource)
    fail(test, "unable to parse the residual-declaration fixture");
  ::fabric::SwitchOp declaration =
      *declarationSource->getOps<::fabric::SwitchOp>().begin();
  rejectMutation("canonical Module payload retains a named declaration",
                 [&](::fabric::ModuleOp canonicalRoot) {
                   mlir::Operation *clone = declaration->clone();
                   mlir::Operation *terminator =
                       canonicalRoot.getBody().front().getTerminator();
                   terminator->getBlock()->getOperations().insert(
                       terminator->getIterator(), clone);
                 });
}

void unregisteredDiscardableAttributesAreRejected() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @unknown_attribute(%value: !fabric.bits<8>)
          -> !fabric.bits<8> {
        %fifo = fabric.fifo %value [max_depth = 2, bypassable = true]
            : !fabric.bits<8>
        fabric.yield %fifo : !fabric.bits<8>
      }
    }
  )mlir",
                                                        &context());
  if (!source)
    fail(test, "unable to parse the discardable-attribute fixture");
  loom::fabric::FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), store));
  loom::fabric::DecodedFabricArtifact decoded =
      take(test, loom::fabric::decodeFabricArtifactEnvelope(
                     finalized.canonicalBytes().bytes()));
  auto shadowParsed =
      take(test, loom::fabric::detail::parseFabricBytecodeModule(
                     decoded.canonicalMlirBytecode));
  ::fabric::ModuleOp shadowRoot = root(test, shadowParsed.module.get());
  mlir::Operation *shadowedYield = shadowRoot.getBody().front().getTerminator();
  shadowedYield->setDiscardableAttr(
      "declared_types",
      mlir::StringAttr::get(shadowRoot.getContext(), "not-an-array"));
  llvm::Error shadowError =
      loom::fabric::detail::validateCanonicalFabricModulePayload(shadowRoot);
  if (!shadowError)
    fail(test, "accepted a discardable shadow of an inherent property");
  const std::string shadowMessage = llvm::toString(std::move(shadowError));
  if (!llvm::StringRef(shadowMessage).contains("discardable attribute"))
    fail(test, shadowMessage);

  auto parsed = take(test, loom::fabric::detail::parseFabricBytecodeModule(
                               decoded.canonicalMlirBytecode));
  ::fabric::ModuleOp canonicalRoot = root(test, parsed.module.get());
  canonicalRoot.getBody().front().getTerminator()->setAttr(
      "review_marker", mlir::BoolAttr::get(canonicalRoot.getContext(), true));

  decoded.canonicalMlirBytecode = take(
      test,
      loom::fabric::detail::writeCanonicalFabricBytecode(parsed.module.get()));
  loom::CanonicalSemanticBytes malformed =
      take(test, loom::fabric::encodeFabricArtifactEnvelope(
                     loom::fabric::FabricRootKind::Module, {},
                     decoded.canonicalMlirBytecode));
  const loom::ArtifactIdentity identity =
      take(test, store.put(loom::fabric::fabricArtifactSchema, malformed));
  expectRejected(test,
                 loom::fabric::importEntireFabricRoot(
                     {loom::fabric::fabricArtifactSchema.identity.str(),
                      loom::fabric::fabricArtifactSchema.version, identity},
                     store),
                 "canonical Module payload has an unregistered discardable "
                 "attribute");

  loom::fabric::DecodedFabricArtifact supplementalDecoded =
      take(test, loom::fabric::decodeFabricArtifactEnvelope(
                     finalized.canonicalBytes().bytes()));
  auto malformedSupplemental =
      take(test, loom::fabric::detail::parseFabricBytecodeModule(
                     supplementalDecoded.canonicalMlirBytecode));
  ::fabric::ModuleOp supplementalRoot =
      root(test, malformedSupplemental.module.get());
  supplementalRoot.getBody().front().getTerminator()->setDiscardableAttr(
      ::fabric::kResourceContractRecordAttrName,
      mlir::BoolAttr::get(supplementalRoot.getContext(), true));
  const std::vector<std::uint8_t> supplementalBytecode =
      take(test, loom::fabric::detail::writeCanonicalFabricBytecode(
                     malformedSupplemental.module.get()));
  loom::CanonicalSemanticBytes supplementalEnvelope =
      take(test,
           loom::fabric::encodeFabricArtifactEnvelope(
               loom::fabric::FabricRootKind::Module, {}, supplementalBytecode));
  const loom::ArtifactIdentity supplementalIdentity =
      take(test,
           store.put(loom::fabric::fabricArtifactSchema, supplementalEnvelope));
  expectRejected(
      test,
      loom::fabric::importEntireFabricRoot(
          {loom::fabric::fabricArtifactSchema.identity.str(),
           loom::fabric::fabricArtifactSchema.version, supplementalIdentity},
          store),
      "canonical Module payload has an unregistered discardable "
      "attribute");
}

void redundantYieldDeclarationsAreCanonicalizedAndRejected() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @yield_default(%value: !fabric.bits<8>)
          -> !fabric.bits<8> {
        %fifo = fabric.fifo %value [max_depth = 2, bypassable = true]
            : !fabric.bits<8>
        fabric.yield %fifo : !fabric.bits<8>
      }
    }
  )mlir",
                                                        &context());
  if (!source)
    fail(test, "unable to parse the yield-default fixture");
  ::fabric::YieldOp authoredYield = mlir::cast<::fabric::YieldOp>(
      root(test, *source).getBody().front().getTerminator());
  authoredYield.setDeclaredTypesAttr(mlir::ArrayAttr::get(
      source->getContext(),
      {mlir::TypeAttr::get(authoredYield.getValues().front().getType())}));

  loom::fabric::FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root(test, *source), store));
  loom::fabric::DecodedFabricArtifact decoded =
      take(test, loom::fabric::decodeFabricArtifactEnvelope(
                     finalized.canonicalBytes().bytes()));
  auto parsed = take(test, loom::fabric::detail::parseFabricBytecodeModule(
                               decoded.canonicalMlirBytecode));
  ::fabric::ModuleOp canonicalRoot = root(test, parsed.module.get());
  ::fabric::YieldOp canonicalYield = mlir::cast<::fabric::YieldOp>(
      canonicalRoot.getBody().front().getTerminator());
  if (canonicalYield.getDeclaredTypesAttr())
    fail(test, "finalizer retained a redundant declared_types property");

  canonicalYield.setDeclaredTypesAttr(mlir::ArrayAttr::get(
      canonicalRoot.getContext(),
      {mlir::TypeAttr::get(canonicalYield.getValues().front().getType())}));
  decoded.canonicalMlirBytecode = take(
      test,
      loom::fabric::detail::writeCanonicalFabricBytecode(parsed.module.get()));
  loom::CanonicalSemanticBytes malformed =
      take(test, loom::fabric::encodeFabricArtifactEnvelope(
                     loom::fabric::FabricRootKind::Module, {},
                     decoded.canonicalMlirBytecode));
  const loom::ArtifactIdentity identity =
      take(test, store.put(loom::fabric::fabricArtifactSchema, malformed));
  expectRejected(test,
                 loom::fabric::importEntireFabricRoot(
                     {loom::fabric::fabricArtifactSchema.identity.str(),
                      loom::fabric::fabricArtifactSchema.version, identity},
                     store),
                 "fabric.yield has redundant declared_types");
}

void memoryBoundaryConnectionCannotCrossModuleSlots() {
  const llvm::StringRef test = __func__;
  using namespace loom::fabric;
  const FabricModuleTemplateRef module(0);
  const FabricMemoryOccurrenceRef memory(1);
  const auto boundary = FabricModuleDomainMemberRef::of(
      FabricModuleBoundaryEndpointRef{module, FabricPortDirection::Input, 0});
  const auto internal = FabricModuleDomainMemberRef::of(
      take(test, FabricModulePhysicalOwnerRef::create(memory)));
  const FabricModuleDomainSlotRef firstClock{module,
                                             FabricClockResetKind::Clock, 0};
  const FabricModuleDomainSlotRef secondClock{module,
                                              FabricClockResetKind::Clock, 1};
  const FabricModuleDomainSlotRef reset{module, FabricClockResetKind::Reset, 0};

  std::vector<std::uint8_t> identityBytes(loom::ArtifactIdentity::byteSize, 0);
  detail::FabricArtifactViewData data(
      take(test, loom::ArtifactIdentity::fromBytes(identityBytes)),
      FabricRootKind::Module);
  data.entities.resize(2);
  data.entities[0].kind = FabricEntityKind::FabricModuleTemplate;
  data.entities[0].owner.inventoryCounts = detail::emptyFabricInventories();
  data.entities[0].moduleBoundaryInputs.push_back(
      {FabricSpatialAttachmentEndpointRef::Plane::Memory, 0, {}});
  data.entities[1].kind = FabricEntityKind::FabricMemoryOccurrence;
  data.entities[1].owner.inventoryCounts = detail::emptyFabricInventories();
  data.entities[1].owner.memoryEndpoints.push_back(
      {FabricMemoryEndpointRole::Manager, {}});
  data.entities[1].memoryConnectivity =
      take(test, ::fabric::MemoryConnectivityContractRecord::create({}));
  data.moduleBoundaryMemoryAttachments.push_back(
      {{module, FabricPortDirection::Input, 0},
       {FabricMemoryEndpointOwnerRef::of(memory), 0}});
  data.moduleDomainSlots = {firstClock, secondClock, reset};
  data.moduleDomainAssignments = {{boundary, firstClock},
                                  {boundary, reset},
                                  {internal, secondClock},
                                  {internal, reset}};

  auto artifact = take(test, detail::buildFabricArtifactView(std::move(data)));
  auto view = take(test, requireModuleRoot(artifact));
  llvm::Error error = validateModuleClockReset(view);
  if (!error)
    fail(test, "accepted a cross-slot Module memory attachment");
  const std::string message = llvm::toString(std::move(error));
  if (!llvm::StringRef(message).contains(
          "crosses symbolic Clock or Reset slots"))
    fail(test, message);
}

} // namespace

int main() {
  nonCanonicalSlotPermutationIsRejected();
  nonCanonicalFuGraphOrderIsRejected();
  canonicalFuDefinitionOrdinalsSurviveStrictImport();
  unregisteredDiscardableAttributesAreRejected();
  redundantYieldDeclarationsAreCanonicalizedAndRejected();
  fabricOpSchemaInventoryIsCanonicalAndStrict();
  missingCanonicalFuCapabilityCarrierIsRejected();
  nonCanonicalModuleGraphOrderIsRejected();
  uncoordinatedStoredEntityIdPermutationIsRejected();
  derivedIdentifiersOnNonCarriersAreRejected();
  canonicalRelationsOnNonCarriersAreRejected();
  authoringStateAndDeclarationsAreRejected();
  memoryBoundaryConnectionCannotCrossModuleSlots();
  return EXIT_SUCCESS;
}
