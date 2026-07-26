#include "ADG/Builder.h"

#include "Common/ArtifactStore.h"
#include "Fabric/IR/MemoryOperationPort.h"
#include "Fabric/Identity/FabricFuCapabilityTemplate.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <utility>

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
                 llvm::StringRef diagnostic) {
  if (!error)
    fail(test, "accepted invalid ADG authoring");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(diagnostic), message);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef diagnostic) {
  if (value)
    fail(test, "accepted invalid ADG authoring");
  expectError(test, value.takeError(), diagnostic);
}

class TemporaryDirectory {
public:
  explicit TemporaryDirectory(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> path;
    if (std::error_code error =
            llvm::sys::fs::createUniqueDirectory("loom-adg-builder", path))
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

using loom::adg::BoundarySpec;
using loom::adg::DesignBuilder;
using loom::adg::FifoSpec;
using loom::adg::FuConfigurationMode;
using loom::adg::FuSpec;
using loom::adg::FuValue;
using loom::adg::MemorySpec;
using loom::adg::OperationCapabilitySpec;
using loom::adg::PeSpec;
using loom::adg::PortType;
using loom::adg::SpatialValue;
using loom::adg::SwitchSpec;
using loom::adg::TemporalPeParameters;

std::uint64_t uniqueEntity(llvm::StringRef test,
                           const loom::fabric::FabricArtifactView &view,
                           loom::fabric::FabricEntityKind expectedKind) {
  std::optional<std::uint64_t> result;
  for (std::uint64_t id = 0;; ++id) {
    std::optional<loom::fabric::FabricEntityKind> kind = view.entityKind(id);
    if (!kind)
      break;
    if (*kind != expectedKind)
      continue;
    if (result)
      fail(test, "finalized root contains duplicate expected entities");
    result = id;
  }
  if (!result)
    fail(test, "finalized root contains no expected entity");
  return *result;
}

loom::fabric::FabricFuTemplateRef
uniqueFuTemplate(llvm::StringRef test,
                 const loom::fabric::FabricArtifactView &view) {
  return loom::fabric::FabricFuTemplateRef(uniqueEntity(
      test, view, loom::fabric::FabricEntityKind::FabricFuTemplate));
}

OperationCapabilitySpec
integerCapability(::fabric::ImplementationFamilyId family,
                  ::dataflow::OperationSchemaId operation,
                  const PortType &outputType) {
  return OperationCapabilitySpec{
      family,
      ::fabric::ScalarIntegerParams{
          ::fabric::IntegerWidthSet::get({::fabric::IntegerWidth::I32})},
      {operation},
      {outputType}};
}

::fabric::UnsignedDomain singleton(std::uint64_t value) {
  return take("memory singleton",
              ::fabric::UnsignedDomain::fromCanonical({{value, value}}));
}

::fabric::MemoryOperationPortDeclaration loadPortDeclaration() {
  ::fabric::ResourceContractDeclaration resource;
  resource.states = {::fabric::ResourceStateDeclaration{
      ::fabric::StateKey(0),
      {{::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1),
        ::fabric::CapacityUnits(0)}}}};
  resource.requesters = {::fabric::RequesterKey(0)};
  resource.eligibilityCount = 1;
  resource.eventCount = 1;
  resource.timingContracts = {{::fabric::TimingContractKey(0), {0}}};
  resource.usePatterns = {{::fabric::UsePatternKey(0),
                           ::fabric::RequesterKey(0),
                           ::fabric::EligibilityKey(0),
                           ::fabric::EventKey(0),
                           ::fabric::EventKey(0),
                           std::nullopt,
                           ::fabric::TimingContractKey(0),
                           {},
                           {{{}}}}};

  auto alignment =
      take("memory alignment",
           ::fabric::AlignmentDomain::create(
               take("memory alignment range",
                    ::fabric::UnsignedDomain::fromCanonical({{0, 63}}))));
  auto read = take(
      "memory read semantics",
      ::fabric::ClosedEnumDomain<::fabric::ReadSubwordSemantics>::fromCanonical(
          {::fabric::ReadSubwordSemantics::Exact}));
  auto write =
      take("memory write semantics",
           ::fabric::ClosedEnumDomain<::fabric::WriteSubwordSemantics>::
               fromCanonical({::fabric::WriteSubwordSemantics::NotApplicable}));
  auto access =
      take("memory access class",
           ::fabric::MemoryAccessClass::create(
               ::dataflow::semantics::MemoryAccessForm::Element, singleton(32),
               singleton(1),
               {{::dataflow::semantics::MemoryMaskForm::Absent,
                 ::fabric::InactiveLaneSemantics::NotApplicable}},
               std::move(alignment), std::move(read), std::move(write)));
  auto accessDomain = take(
      "memory access domain",
      ::fabric::ParameterizedMemoryAccessDomain::create({std::move(access)}));
  ::fabric::MemoryActorContractClause plain =
      ::fabric::LoadStorePlainContractClause{{false}};
  auto actorDomain =
      take("memory actor domain",
           ::fabric::MemoryActorContractDomain::create(
               ::dataflow::OperationSchemaId::DataflowLoad, {plain}));

  return {{0, 1, 2, 3},
          take("memory resource contract",
               ::fabric::ResourceContract::create(std::move(resource))),
          {{::fabric::MemoryPortTransactionProjection::Direct}},
          {{std::move(actorDomain),
            {{::dataflow::semantics::ServiceValueRole::Address, 0},
             {::dataflow::semantics::ServiceValueRole::Control, 1},
             {::dataflow::semantics::ServiceValueRole::Data, 2},
             {::dataflow::semantics::ServiceValueRole::Completion, 3}},
            std::move(accessDomain),
            {::fabric::UsePatternKey(0)}}}};
}

void regularAndIrregularSpatialCoresFinalize() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);

  const PortType bits4 = take(test, PortType::bits(4));
  const PortType bits32 = take(test, PortType::bits(32));
  const PortType bits64 = take(test, PortType::bits(64));
  const PortType tagged32x4 = take(test, PortType::taggedBits(32, 4));

  auto regular = take(
      test, design.createSpatialCore("regular", {bits32, bits4}, {tagged32x4}));
  SpatialValue regularData = take(test, regular.input(0));
  SpatialValue regularTag = take(test, regular.input(1));
  auto regularBoundary = take(
      test, regular.addBoundary({regularData, regularTag},
                                BoundarySpec::s2t(bits32, bits4, tagged32x4)));
  SpatialValue regularQueued =
      take(test, regular.addFifo(regularBoundary.front(),
                                 FifoSpec{tagged32x4, 2, true}));
  if (llvm::Error error = regular.close({regularQueued}))
    fail(test, llvm::toString(std::move(error)));

  auto irregular =
      take(test,
           design.createSpatialCore("irregular", {bits64, bits32, bits4, bits4},
                                    {tagged32x4, tagged32x4}));
  SpatialValue irregularData = take(test, irregular.input(0));
  SpatialValue alternateData = take(test, irregular.input(1));
  SpatialValue irregularTag0 = take(test, irregular.input(2));
  SpatialValue irregularTag1 = take(test, irregular.input(3));
  SpatialValue narrowed =
      take(test, irregular.addFifo(irregularData, FifoSpec{bits32, 3, false}));
  auto switched =
      take(test, irregular.addSwitch({narrowed, alternateData},
                                     SwitchSpec::spatial({bits32, bits32},
                                                         {bits32, bits32},
                                                         {{0, 1}, {1}})));
  auto irregularBoundary0 =
      take(test,
           irregular.addBoundary({switched[0], irregularTag0},
                                 BoundarySpec::s2t(bits32, bits4, tagged32x4)));
  auto irregularBoundary1 =
      take(test,
           irregular.addBoundary({switched[1], irregularTag1},
                                 BoundarySpec::s2t(bits32, bits4, tagged32x4)));
  if (llvm::Error error = irregular.close(
          {irregularBoundary0.front(), irregularBoundary1.front()}))
    fail(test, llvm::toString(std::move(error)));

  loom::adg::FinalizedFabricDesign finalized =
      take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 2,
          "finalized design did not contain both SpatialCore roots");
  for (const loom::fabric::FinalizedFabricRoot &root : finalized.roots()) {
    require(test,
            root.view().rootKind() == loom::fabric::FabricRootKind::Module,
            "SpatialCore finalized with a non-Module root kind");
    require(test, !root.view().admittedTraversals().empty(),
            "SpatialCore lost its physical traversal inventory");
  }
}

void foreignHandlesAndIncompleteRootsFailClosed() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits32 = take(test, PortType::bits(32));
  expectError(test, PortType::memory({-1}, bits32), "invalid extent");

  auto first =
      take(test, design.createSpatialCore("first", {bits32}, {bits32}));
  auto second =
      take(test, design.createSpatialCore("second", {bits32}, {bits32}));
  SpatialValue foreign = take(test, first.input(0));
  expectError(test, second.addFifo(foreign, FifoSpec{bits32, 1, false}),
              "foreign SpatialValue");

  if (llvm::Error error = first.close({foreign}))
    fail(test, llvm::toString(std::move(error)));
  expectError(test, std::move(design).finalize(),
              "SpatialCore 'second' is not closed");
}

void typedPeFuGraphsFinalize() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);

  const PortType bits32 = take(test, PortType::bits(32));
  const PortType tagged32x4 = take(test, PortType::taggedBits(32, 4));

  auto spatial = take(
      test, design.createSpatialCore("spatial-pe", {bits32, bits32}, {bits32}));
  SpatialValue spatialA = take(test, spatial.input(0));
  SpatialValue spatialB = take(test, spatial.input(1));
  auto spatialPe =
      take(test, spatial.addPe({spatialA, spatialB},
                               PeSpec::spatial({bits32, bits32}, {bits32})));
  auto spatialFu =
      take(test, spatialPe.addFu({take(test, spatialPe.input(0)),
                                  take(test, spatialPe.input(1))},
                                 FuSpec{{bits32, bits32}, {bits32}}));
  FuValue fuA = take(test, spatialFu.input(0));
  FuValue fuB = take(test, spatialFu.input(1));
  auto aLanes = take(test, spatialFu.addDemux(fuA, 2));
  auto bLanes = take(test, spatialFu.addDemux(fuB, 2));
  auto sum =
      take(test, spatialFu.addOperation(
                     {aLanes[0], bLanes[0]},
                     integerCapability(
                         ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                         ::dataflow::OperationSchemaId::ArithAddI, bits32)));
  auto product = take(
      test, spatialFu.addOperation(
                {aLanes[1], bLanes[1]},
                integerCapability(
                    ::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
                    ::dataflow::OperationSchemaId::ArithMulI, bits32)));
  FuValue selected =
      take(test, spatialFu.addMux({sum.front(), product.front()}));
  if (llvm::Error error = spatialFu.close({selected}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatialPe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatial.close({take(test, spatialPe.output(0))}))
    fail(test, llvm::toString(std::move(error)));

  auto temporal =
      take(test, design.createSpatialCore(
                     "temporal-pe", {tagged32x4, tagged32x4}, {tagged32x4}));
  auto temporalPe = take(
      test, temporal.addPe(
                {take(test, temporal.input(0)), take(test, temporal.input(1))},
                PeSpec::temporal({bits32, bits32}, {tagged32x4},
                                 TemporalPeParameters{
                                     2, FuConfigurationMode::PerFu,
                                     ::fabric::OperandBufferMode::PerInputPort,
                                     2, std::nullopt})));
  auto temporalFu =
      take(test, temporalPe.addFu({take(test, temporalPe.input(0)),
                                   take(test, temporalPe.input(1))},
                                  FuSpec{{bits32, bits32}, {bits32}}));
  auto temporalSum = take(
      test,
      temporalFu.addOperation(
          {take(test, temporalFu.input(0)), take(test, temporalFu.input(1))},
          integerCapability(
              ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
              ::dataflow::OperationSchemaId::ArithAddI, bits32)));
  if (llvm::Error error = temporalFu.close({temporalSum.front()}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = temporalPe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = temporal.close({take(test, temporalPe.output(0))}))
    fail(test, llvm::toString(std::move(error)));

  loom::adg::FinalizedFabricDesign finalized =
      take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 2,
          "typed PE/FU authoring did not publish both roots");

  const auto spatialTemplates =
      finalized.roots()[0].view().fuCapabilityTemplates(
          uniqueFuTemplate(test, finalized.roots()[0].view()));
  require(test, spatialTemplates.size() == 2,
          "explicit add-or-multiply routing did not form two FU templates");
  const auto temporalTemplates =
      finalized.roots()[1].view().fuCapabilityTemplates(
          uniqueFuTemplate(test, finalized.roots()[1].view()));
  require(test, temporalTemplates.size() == 1,
          "temporal PE operation did not form one FU template");
  const loom::fabric::FabricPeOccurrenceRef temporalPeRef(
      uniqueEntity(test, finalized.roots()[1].view(),
                   loom::fabric::FabricEntityKind::FabricPeOccurrence));
  const ::fabric::ResourceContract *operandBuffer =
      finalized.roots()[1].view().resourceContract(
          loom::fabric::FabricInventoryOwnerRef::of(temporalPeRef));
  require(test,
          operandBuffer && operandBuffer->stateCount() != 0 &&
              operandBuffer->usePatternCount() != 0,
          "temporal PE lost its operand-buffer resource contract");
}

void typedMemoryEngineFinalizes() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);

  const PortType bits0 = take(test, PortType::bits(0));
  const PortType bits32 = take(test, PortType::bits(32));
  const PortType memory32 =
      take(test, PortType::memory({PortType::kDynamicExtent}, bits32));
  auto spatial = take(test, design.createSpatialCore("memory-engine",
                                                     {memory32, bits32, bits0},
                                                     {bits32, bits0}));
  auto outputs = take(
      test, spatial.addMemory(
                {take(test, spatial.input(0)), take(test, spatial.input(1)),
                 take(test, spatial.input(2))},
                MemorySpec::spatial({memory32, bits32, bits0}, {bits32, bits0},
                                    {0}, {}, {loadPortDeclaration()})));
  if (llvm::Error error = spatial.close(outputs))
    fail(test, llvm::toString(std::move(error)));

  loom::adg::FinalizedFabricDesign finalized =
      take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "memory design did not publish one SpatialCore root");
  const auto &view = finalized.roots().front().view();
  const loom::fabric::FabricMemoryOccurrenceRef memory(uniqueEntity(
      test, view, loom::fabric::FabricEntityKind::FabricMemoryOccurrence));
  auto ports = view.memoryOperationPorts(memory);
  require(test,
          ports.size() == 1 && view.memoryOperationPort(ports.front()) &&
              view.memoryCapabilityAlternative({ports.front(), 0}),
          "typed memory capability was not preserved by Fabric finalization");
}

} // namespace

int main() {
  regularAndIrregularSpatialCoresFinalize();
  foreignHandlesAndIncompleteRootsFailClosed();
  typedPeFuGraphsFinalize();
  typedMemoryEngineFinalizes();
  return EXIT_SUCCESS;
}
