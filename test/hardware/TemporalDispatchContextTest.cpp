// Anchors the Temporal PE dispatch-context and result-egress contracts of
// docs/spec-rtl-lowering.md with the exact generated RTL of three Temporal
// PEs whose resident realizations live in nonzero or several contexts:
//  - a dataflow.invariant whose Init case consumes the Init head alone and
//    whose Replay and Close cases consume the phase head alone, proving that a
//    stateful transition fires on the partial operand tuple its schema case
//    consumes and that the context's state bank persists across cases;
//  - a dataflow.gate whose ContinueTrue case publishes its phase and value
//    results atomically through two arbitrated PE output ports, proving that
//    egress readiness is observable before either result asserts valid;
//  - an add and a multiply of one FU behind demux/mux selectors, resident in
//    two contexts and routed to the same PE output port, proving that an FU
//    boundary output shared by several held results retires exactly one
//    result per handoff instead of dropping the other.
// The emitted testbenches send the heads one at a time and expect exactly the
// schema-owned publications.

#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/LoopGate.h"
#include "Hardware/RTL/Providers/LoopInvariant.h"
#include "Hardware/RTL/Providers/ScalarIntegerAddSub.h"
#include "Hardware/RTL/Providers/ScalarIntegerMultiply.h"
#include "Hardware/RTL/Specialization.h"

#include "ConfigurationABITestSupport.h"
#include "ConfigurationTransportTestSupport.h"
#include "PortableProviderTestSupport.h"

#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/ImplementationFamily.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/Identity/FabricSemanticFieldRelation.h"
#include "Fabric/Identity/FabricTemporalPeConfiguration.h"
#include "Hardware/Implementation/HardwareImplementation.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::fabric::FinalizedFabricRoot;
using loom::hardware::FinalizedConfigurationABI;

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

/// The resident realization a fixture hosts. The loop fixtures place one
/// actor in context 1 of a two-context PE (FU input 0 is the phase and FU
/// input 1 the value; an invariant publishes one value result, a gate
/// publishes its phase on FU output 0 and its value on FU output 1). The
/// shared-output fixture places an add in context 0 and a multiply in
/// context 1 of one FU whose single boundary output both results leave.
enum class Fixture { Invariant, Gate, SharedOutput };

llvm::StringRef fixtureName(Fixture fixture) {
  switch (fixture) {
  case Fixture::Invariant:
    return "invariant";
  case Fixture::Gate:
    return "gate";
  case Fixture::SharedOutput:
    return "shared_output";
  }
  return "";
}

std::uint32_t resultCount(Fixture fixture) {
  return fixture == Fixture::Gate ? 2 : 1;
}

/// One active instruction row of a fixture: the context it occupies, the FU
/// capability template that context selects, and the tags its operands (on
/// input ports 0 and 1) and results (on the output ports) carry.
struct ResidentRow final {
  std::uint32_t context = 0;
  std::uint32_t templateOrdinal = 0;
  std::uint32_t operandTag = 0;
  std::uint32_t resultTag = 0;
};

struct FixtureModule final {
  FinalizedFabricRoot root;
  std::vector<ResidentRow> rows;
};

/// The implementation family of the operation whose FU hosts the fixture's
/// resident rows.
::fabric::ImplementationFamilyId hostFamily(Fixture fixture) {
  switch (fixture) {
  case Fixture::Invariant:
    return ::fabric::ImplementationFamilyId::LoopInvariant;
  case Fixture::Gate:
    return ::fabric::ImplementationFamilyId::LoopGate;
  case Fixture::SharedOutput:
    return ::fabric::ImplementationFamilyId::ScalarIntegerAddSub;
  }
  return ::fabric::ImplementationFamilyId::LoopInvariant;
}

std::size_t fuCount(Fixture fixture) {
  return fixture == Fixture::Gate ? 2 : 1;
}

std::size_t operationCount(Fixture fixture) {
  return fixture == Fixture::Invariant ? 1 : 2;
}

/// The unique FU occurrence whose capability templates activate an operation
/// of the given family.
loom::fabric::FabricFuOccurrenceRef
hostFu(llvm::StringRef test, const loom::fabric::FabricArtifactView &view,
       ::fabric::ImplementationFamilyId family) {
  std::optional<loom::fabric::FabricFuOccurrenceRef> result;
  for (const auto fu : view.fuOccurrences()) {
    const auto definition = view.fuTemplateOf(fu);
    require(test, definition.has_value(), "fixture FU has no definition");
    for (const auto &record : view.fuCapabilityTemplates(*definition))
      for (const auto &node : record.activeNodes) {
        if (node.node != loom::fabric::FabricFuNodeKind::Op)
          continue;
        const auto *capability = view.resolvedFabricOpCapability(node);
        if (!capability || capability->implementationFamily != family)
          continue;
        require(test, !result || *result == fu,
                "fixture family is hosted by several FUs");
        result = fu;
      }
  }
  require(test, result.has_value(), "fixture family is hosted by no FU");
  return *result;
}

loom::adg::OperationCapabilitySpec
integerCapability(llvm::StringRef test, ::fabric::ImplementationFamilyId family,
                  ::dataflow::OperationSchemaId operation,
                  const loom::adg::PortType &outputType) {
  const auto width = llvm::find_if(
      ::fabric::integerWidthDomain, [&](::fabric::IntegerWidth candidate) {
        return ::fabric::getBitWidth(candidate) == outputType.width();
      });
  require(test, width != ::fabric::integerWidthDomain.end(),
          "fixture port has no scalar integer width");
  return loom::adg::OperationCapabilitySpec{
      family,
      ::fabric::ScalarIntegerParams{::fabric::IntegerWidthSet::get({*width})},
      {operation},
      {outputType},
      ::fabric::oneCycleElasticOperationResourceContract()};
}

/// One Temporal PE with two resident contexts whose single FU hosts the
/// fixture's operations.
FixtureModule makeFixtureModule(llvm::StringRef test,
                                const ArtifactStore &store, Fixture fixture) {
  using namespace loom::adg;
  DesignBuilder design(store);
  const PortType bits1 = take(test, PortType::bits(1));
  const PortType bits8 = take(test, PortType::bits(8));
  const PortType tagged8x2 = take(test, PortType::taggedBits(8, 2));
  const std::vector<PortType> peOutputs(resultCount(fixture), tagged8x2);
  auto spatial = take(
      test, design.createSpatialCore(
                ("temporal-dispatch-context-" + fixtureName(fixture)).str(),
                {tagged8x2, tagged8x2}, peOutputs));
  auto pe = take(
      test,
      spatial.addPe(
          {take(test, spatial.input(0)), take(test, spatial.input(1))},
          PeSpec::temporal({bits8, bits8}, peOutputs,
                           TemporalPeParameters{
                               2, FuConfigurationMode::PerInstruction,
                               ::fabric::OperandBufferMode::PerInstruction, 1,
                               std::nullopt})));
  std::vector<ResidentRow> rows;
  std::vector<FuValue> fuOutputs;
  std::optional<FuBuilder> fu;
  std::optional<FuCapabilityTemplateHandle> sumTemplate;
  std::optional<FuCapabilityTemplateHandle> productTemplate;
  if (fixture == Fixture::SharedOutput) {
    fu.emplace(take(test, pe.addFu({take(test, pe.input(0)),
                                    take(test, pe.input(1))},
                                   FuSpec{{bits8, bits8}, {bits8}})));
    auto aRoutes = take(test, fu->addDemux(take(test, fu->input(0)), 2));
    auto bRoutes = take(test, fu->addDemux(take(test, fu->input(1)), 2));
    auto sum = take(
        test, fu->addOperation(
                  {take(test, aRoutes.output(0)), take(test, bRoutes.output(0))},
                  integerCapability(
                      test, ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                      ::dataflow::OperationSchemaId::ArithAddI, bits8)));
    auto product = take(
        test, fu->addOperation(
                  {take(test, aRoutes.output(1)), take(test, bRoutes.output(1))},
                  integerCapability(
                      test,
                      ::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
                      ::dataflow::OperationSchemaId::ArithMulI, bits8)));
    auto resultMux = take(test, fu->addMux({take(test, sum.output(0)),
                                            take(test, product.output(0))}));
    sumTemplate = take(test, fu->addCapabilityTemplateWithHandle(
                                 FuCapabilityTemplateSpec{
                                     {sum},
                                     {{aRoutes, 0}, {bRoutes, 0}, {resultMux, 0}}}));
    productTemplate = take(
        test, fu->addCapabilityTemplateWithHandle(FuCapabilityTemplateSpec{
                  {product}, {{aRoutes, 1}, {bRoutes, 1}, {resultMux, 1}}}));
    fuOutputs.push_back(take(test, resultMux.output(0)));
  } else {
    // FU boundary outputs carry the PE data width; the gate's one-bit phase
    // result widens at its FU output like any narrower operation result.
    const std::vector<PortType> operationOutputs =
        fixture == Fixture::Gate ? std::vector<PortType>{bits1, bits8}
                                 : std::vector<PortType>{bits8};
    fu.emplace(take(
        test, pe.addFu({take(test, pe.input(0)), take(test, pe.input(1))},
                       FuSpec{{bits1, bits8},
                              std::vector<PortType>(resultCount(fixture),
                                                    bits8)})));
    auto operation = take(
        test,
        fu->addOperation(
            {take(test, fu->input(0)), take(test, fu->input(1))},
            fixture == Fixture::Gate
                ? OperationCapabilitySpec{
                      ::fabric::ImplementationFamilyId::LoopGate,
                      ::fabric::TokenPlaneParams{},
                      {::dataflow::OperationSchemaId::DataflowGate},
                      operationOutputs,
                      ::fabric::loopGateOperationResourceContract()}
                : OperationCapabilitySpec{
                      ::fabric::ImplementationFamilyId::LoopInvariant,
                      ::fabric::TokenPlaneParams{},
                      {::dataflow::OperationSchemaId::DataflowInvariant},
                      operationOutputs,
                      ::fabric::loopInvariantOperationResourceContract()}));
    if (llvm::Error error = fu->addCapabilityTemplate(
            FuCapabilityTemplateSpec{{operation}, {}}))
      fail(test, llvm::toString(std::move(error)));
    for (std::uint32_t ordinal = 0; ordinal != resultCount(fixture); ++ordinal)
      fuOutputs.push_back(take(test, operation.output(ordinal)));
    // The tested realization lives in context 1: phase from port 0 tag 1,
    // value from port 1 tag 1, result k to port k tag 2. The invariant leaves
    // row 0 Unused. The gate also occupies context 0 with a realization on
    // tag 3 that is never fed, so its FU's dispatch keeps rotating over two
    // contexts, and its PE hosts a second FU without resident rows, so the
    // idle-egress pointer has to align with that rotation.
    rows.push_back({1, 0, 1, 2});
    if (fixture == Fixture::Gate) {
      rows.push_back({0, 0, 3, 3});
      auto idleFu = take(
          test, pe.addFu({take(test, pe.input(0)), take(test, pe.input(1))},
                         FuSpec{{bits8, bits8}, {bits8}}));
      auto sum = take(
          test, idleFu.addOperation(
                    {take(test, idleFu.input(0)), take(test, idleFu.input(1))},
                    integerCapability(
                        test,
                        ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                        ::dataflow::OperationSchemaId::ArithAddI, bits8)));
      if (llvm::Error error = idleFu.addCapabilityTemplate(
              FuCapabilityTemplateSpec{{sum}, {}}))
        fail(test, llvm::toString(std::move(error)));
      if (llvm::Error error = idleFu.close({take(test, sum.output(0))}))
        fail(test, llvm::toString(std::move(error)));
    }
  }
  if (llvm::Error error = fu->close(fuOutputs))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  std::vector<SpatialValue> spatialOutputs;
  for (std::uint32_t ordinal = 0; ordinal != resultCount(fixture); ++ordinal)
    spatialOutputs.push_back(take(test, pe.output(ordinal)));
  if (llvm::Error error = spatial.close(spatialOutputs))
    fail(test, llvm::toString(std::move(error)));
  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "dispatch-context fixture did not finalize one Module");
  if (fixture == Fixture::SharedOutput) {
    // The add lives in context 0 (operands tag 1, result tag 2) and the
    // multiply in context 1 (operands tag 3, result tag 3); both results
    // leave through PE output port 0.
    const auto templateOrdinal =
        [&](const loom::adg::FuCapabilityTemplateHandle &handle) {
          return static_cast<std::uint32_t>(
              take(test, finalized.resolve(handle)).entity.ordinal);
        };
    rows.push_back({0, templateOrdinal(*sumTemplate), 1, 2});
    rows.push_back({1, templateOrdinal(*productTemplate), 3, 3});
  }
  return FixtureModule{std::move(finalized.roots().front()), std::move(rows)};
}

loom::fabric::FabricPhysicalConfigurationFieldRef qualifyConfigurationField(
    llvm::StringRef test, loom::fabric::SpatialCoreOccurrenceRef spatialCore,
    const loom::fabric::FabricSemanticConfigFieldRef &field) {
  auto target =
      take(test, loom::fabric::FabricModulePhysicalTargetRef::create(field));
  return take(test, loom::fabric::FabricPhysicalConfigurationFieldRef::create(
                        loom::fabric::SpatialCoreInternalOccurrenceRef{
                            spatialCore, std::move(target)}));
}

const loom::hardware::ProgrammingUnit *findProgrammingOwner(
    llvm::StringRef test, const loom::hardware::ConfigurationABI &abi,
    const loom::fabric::FabricPhysicalConfigurationSlotRef &slot) {
  const loom::hardware::ProgrammingUnit *result = nullptr;
  for (const auto &unit : abi.programmingUnits())
    for (const auto &field : unit.fields)
      if (field.slot == slot) {
        require(test, result == nullptr,
                "configuration field has duplicate programming owners");
        result = &unit;
      }
  require(test, result != nullptr,
          "configuration field has no programming owner");
  return result;
}

struct DispatchContextArtifact final {
  Fixture fixture = Fixture::Invariant;
  std::string systemVerilog;
  loom::hardware::test::PortableConfigurationTarget target;
  std::vector<std::uint8_t> image;
};

DispatchContextArtifact buildArtifact(const std::filesystem::path &root,
                                      Fixture fixture) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FixtureModule module = makeFixtureModule(test, store, fixture);
  FinalizedFabricRoot system =
      take(test, loom::hardware::test::makeSingleSpatialCoreSystem(module.root,
                                                                   store));
  const loom::fabric::SpatialCoreOccurrenceRef spatialCore = take(
      test, loom::hardware::test::requireSingleSpatialCoreOccurrence(system));
  const auto &view = module.root.view();
  require(test,
          view.peOccurrences().size() == 1 &&
              view.fuOccurrences().size() == fuCount(fixture),
          "dispatch-context fixture changed its PE/FU shape");
  const auto pe = view.peOccurrences().front();
  const auto fu = hostFu(test, view, hostFamily(fixture));

  // The Temporal PE carrier is a Direct field; the ABI draft needs its exact
  // bit width like every other direct carrier of the fixture.
  std::vector<loom::hardware::test::ConfigurationFieldEncodingOverride>
      overrides;
  const loom::fabric::FabricInventoryOwnerRef peOwner =
      loom::fabric::FabricInventoryOwnerRef::of(pe);
  const std::uint64_t peFieldCount = view.inventorySize(
      peOwner, loom::fabric::FabricInventoryKind::SemanticConfigField);
  for (std::uint64_t ordinal = 0; ordinal < peFieldCount; ++ordinal) {
    const loom::fabric::FabricSemanticConfigFieldRef field{
        loom::fabric::FabricConfigurationOwnerRef(peOwner), ordinal};
    auto relation = take(
        test, view.semanticFieldRelation(
                  field, *const_cast<mlir::Operation *>(view.canonicalOperation())
                              ->getContext()));
    if (relation.kind() != loom::fabric::FabricSemanticFieldRelationKind::Direct)
      continue;
    const std::uint64_t bitCount = *relation.directEncodedBitCount();
    overrides.push_back({qualifyConfigurationField(test, spatialCore, field),
                         loom::hardware::DirectBitsEncoding{bitCount},
                         std::vector<std::uint8_t>((bitCount + 7) / 8, 0)});
  }
  FinalizedConfigurationABI abi = take(
      test, loom::hardware::finalizeConfigurationABI(
                take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                               system, overrides)),
                store));

  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto skeleton = take(test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                                 context, spatialCore, abi));
  require(test, skeleton.operationLeaves.size() == operationCount(fixture),
          "dispatch-context skeleton changed its operation leaf count");
  loom::hardware::rtl::FabricOperationProviderRegistry providers;
  const auto registerProviders = [&]() -> llvm::Error {
    switch (fixture) {
    case Fixture::Invariant:
      return loom::hardware::rtl::registerPortableLoopInvariantProvider(
          providers);
    case Fixture::Gate:
      if (llvm::Error error =
              loom::hardware::rtl::registerPortableLoopGateProvider(providers))
        return error;
      return loom::hardware::rtl::registerPortableScalarIntegerAddSubProvider(
          providers);
    case Fixture::SharedOutput:
      if (llvm::Error error =
              loom::hardware::rtl::registerPortableScalarIntegerAddSubProvider(
                  providers))
        return error;
      return loom::hardware::rtl::registerPortableScalarIntegerMultiplyProvider(
          providers);
    }
    return llvm::Error::success();
  };
  if (llvm::Error error = registerProviders())
    fail(test, llvm::toString(std::move(error)));
  loom::hardware::ExternalImplementationContractCatalog externalContracts;
  auto conformance = take(
      test, loom::hardware::test::specializeAndExportPortableProvider(
                std::move(skeleton), abi, providers, externalContracts));

  auto schema = take(test, view.temporalPeConfigurationSchema(pe));
  require(test,
          schema.layout().contextCount == 2 &&
              schema.layout().inputPortCount == 2 &&
              schema.layout().outputPortCount == resultCount(fixture),
          "dispatch-context fixture changed its carrier shape");
  const auto routeInput = [&](std::uint32_t port, std::uint32_t tag) {
    return loom::fabric::FabricTemporalPeOperandSelection{
        loom::fabric::FabricTemporalPeSelectorKind::Route,
        loom::fabric::FabricTemporalPeSelectorTarget{
            loom::fabric::FabricTemporalPePortTarget{port}},
        llvm::APInt(schema.layout().tagWidthBits, tag)};
  };
  const auto routeOutput = [&](std::uint32_t port, std::uint32_t tag) {
    return loom::fabric::FabricTemporalPeResultSelection{
        loom::fabric::FabricTemporalPeSelectorKind::Route,
        loom::fabric::FabricTemporalPeSelectorTarget{
            loom::fabric::FabricTemporalPePortTarget{port}},
        llvm::APInt(schema.layout().tagWidthBits, tag)};
  };
  loom::fabric::FabricTemporalPeActive active;
  active.rows.resize(schema.layout().contextCount);
  for (const ResidentRow &resident : module.rows) {
    loom::fabric::FabricTemporalPeInstructionEntry row{
        fu,
        {routeInput(0, resident.operandTag), routeInput(1, resident.operandTag)},
        {}};
    for (std::uint32_t ordinal = 0; ordinal != resultCount(fixture); ++ordinal)
      row.resultSelections.push_back(routeOutput(ordinal, resident.resultTag));
    active.rows[resident.context] = std::move(row);
  }
  auto peSemantic = take(test, schema.encode(active));
  const auto peSlot =
      take(test, loom::fabric::qualifyFabricConfigurationSlot(
                     qualifyConfigurationField(test, spatialCore, schema.field()),
                     loom::fabric::FabricStaticConfigurationResidency{}));
  const loom::hardware::ProgrammingUnit *owner =
      findProgrammingOwner(test, abi.abi(), peSlot);
  std::vector<loom::hardware::SemanticConfigurationValue> values;
  values.push_back(
      {peSlot, std::vector<std::uint8_t>(peSemantic.bytes().begin(),
                                         peSemantic.bytes().end())});

  const loom::fabric::FabricSemanticConfigFieldRef fuField{
      loom::fabric::FabricConfigurationOwnerRef(
          loom::fabric::FabricInventoryOwnerRef::of(fu)),
      0};
  const auto definition = view.fuTemplateOf(fu);
  require(test, definition.has_value(), "dispatch-context FU has no definition");
  for (const ResidentRow &resident : module.rows) {
    auto fuSemantic = take(
        test, loom::fabric::encodeFabricFuConfiguration(
                  view, fuField,
                  loom::fabric::FabricFuCapabilityTemplateRef{
                      *definition, resident.templateOrdinal}));
    const auto fuSlot = take(
        test, loom::fabric::qualifyFabricConfigurationSlot(
                  qualifyConfigurationField(test, spatialCore, fuField),
                  loom::fabric::InstructionContextRef{pe, resident.context}));
    require(test,
            findProgrammingOwner(test, abi.abi(), fuSlot)->id == owner->id,
            "dispatch-context PE and FU fields span programming units");
    values.push_back(
        {fuSlot, std::vector<std::uint8_t>(fuSemantic.bytes().begin(),
                                           fuSemantic.bytes().end())});
  }
  auto operations = take(
      test, loom::hardware::rtl::enumerateFabricPhysicalOperations(
                take(test, loom::fabric::requireSystemRoot(system.view()))));
  require(test, operations.size() == operationCount(fixture),
          "dispatch-context fixture changed its physical operation count");
  for (const auto &operation : operations)
    require(test, operation.capability->configurationFieldSchema.empty(),
            "fixture operation unexpectedly requires operation configuration");

  return DispatchContextArtifact{
      fixture, std::move(conformance.systemVerilog),
      take(test, loom::hardware::test::derivePortableConfigurationTarget(
                     abi, spatialCore, owner->id)),
      take(test, abi.abi().encode(owner->id, values))};
}

void writeArtifacts(const std::filesystem::path &root,
                    const DispatchContextArtifact &artifact) {
  const llvm::StringRef test = __func__;
  const std::string prefix =
      ("temporal_dispatch_context_" + fixtureName(artifact.fixture)).str();
  const bool gate = artifact.fixture == Fixture::Gate;
  const bool sharedOutput = artifact.fixture == Fixture::SharedOutput;
  std::ofstream(root / (prefix + "_module.sv")) << artifact.systemVerilog;
  std::ofstream testbench(root / (prefix + "_testbench.sv"));
  testbench << "\nmodule " << prefix << R"sv(_testbench;
  logic       clock;
  logic       reset;
  logic [7:0] input_0_data;
  logic [1:0] input_0_tag;
  logic       input_0_valid;
  logic       input_0_ready;
  logic [7:0] input_1_data;
  logic [1:0] input_1_tag;
  logic       input_1_valid;
  logic       input_1_ready;
  logic [7:0] output_0_data;
  logic [1:0] output_0_tag;
  logic       output_0_valid;
  logic       output_0_ready;
)sv";
  if (sharedOutput)
    testbench << R"sv(  logic       output_0_ready_base;
  logic       refuse_enabled;
  logic [1:0] refused_tag;
  always_comb output_0_ready =
      output_0_ready_base && !(refuse_enabled && output_0_tag == refused_tag);
)sv";
  if (gate)
    testbench << R"sv(  logic [7:0] output_1_data;
  logic [1:0] output_1_tag;
  logic       output_1_valid;
  logic       output_1_ready;
)sv";
  testbench << loom::hardware::test::portableAxiLiteSignalDeclarations()
            << "\n";
  testbench << R"sv(
  loom_module dut(.*);
  always #5 clock = ~clock;

  task automatic check(bit condition, string message);
    if (!condition)
      $fatal(1, "%s", message);
  endtask

  task automatic send_input_0(input logic [1:0] tag, input logic [7:0] data);
    integer wait_cycles;
    begin
      @(negedge clock);
      input_0_data = data;
      input_0_tag = tag;
      input_0_valid = 1;
      wait_cycles = 0;
      do begin
        @(posedge clock);
        wait_cycles = wait_cycles + 1;
        if (wait_cycles == 32 && !input_0_ready)
          $fatal(1, "Input port 0 handshake timed out");
      end while (!input_0_ready);
      @(negedge clock);
      input_0_valid = 0;
    end
  endtask

  task automatic send_input_1(input logic [1:0] tag, input logic [7:0] data);
    integer wait_cycles;
    begin
      @(negedge clock);
      input_1_data = data;
      input_1_tag = tag;
      input_1_valid = 1;
      wait_cycles = 0;
      do begin
        @(posedge clock);
        wait_cycles = wait_cycles + 1;
        if (wait_cycles == 32 && !input_1_ready)
          $fatal(1, "Input port 1 handshake timed out");
      end while (!input_1_ready);
      @(negedge clock);
      input_1_valid = 0;
    end
  endtask

)sv";
  if (gate)
    testbench << R"sv(  // Waits for the value result and checks whether the phase result is
  // published in the same cycle, as the gate's atomic tuple requires.
  task automatic expect_gate(input bit phase_expected,
                             input logic [7:0] phase_data,
                             input bit value_expected,
                             input logic [7:0] value_data, string message);
    integer wait_cycles;
    begin
      wait_cycles = 0;
      #1;
      while (!(output_0_valid || output_1_valid)) begin
        @(posedge clock);
        #1;
        wait_cycles = wait_cycles + 1;
        if (wait_cycles == 16)
          $fatal(1, "%s", message);
      end
      check(output_0_valid == phase_expected &&
                output_1_valid == value_expected,
            "Gate did not publish its schema-owned result tuple atomically");
      if (phase_expected)
        check(output_0_data == phase_data && output_0_tag == 2'd2,
              "Gate phase result carried the wrong data or configured tag");
      if (value_expected)
        check(output_1_data == value_data && output_1_tag == 2'd2,
              "Gate value result carried the wrong data or configured tag");
      @(posedge clock);
      #1;
      check(!output_0_valid && !output_1_valid,
            "Consumed gate results remained valid");
    end
  endtask

  task automatic expect_silence(string message);
    begin
      repeat (8) begin
        @(posedge clock);
        #1;
        check(!output_0_valid && !output_1_valid, message);
      end
    end
  endtask

)sv";
  else if (sharedOutput)
    testbench << R"sv(  // Samples the output handshake at every clock edge until one held result
  // has retired through the shared FU boundary output; it must carry the
  // expected tag and data.
  task automatic expect_shared_result(input logic [1:0] tag,
                                      input logic [7:0] data, string message);
    integer wait_cycles;
    bit seen;
    begin
      seen = 0;
      wait_cycles = 0;
      while (!seen) begin
        @(posedge clock);
        wait_cycles = wait_cycles + 1;
        if (wait_cycles == 16)
          $fatal(1, "%s", message);
        if (output_0_valid && output_0_ready) begin
          check(output_0_tag == tag && output_0_data == data,
                "Shared FU output published an unexpected token");
          seen = 1;
        end
      end
    end
  endtask

  task automatic expect_silence(string message);
    begin
      repeat (8) begin
        @(posedge clock);
        #1;
        check(!output_0_valid, message);
      end
    end
  endtask

)sv";
  else
    testbench << R"sv(  task automatic expect_result(input logic [7:0] data, string message);
    integer wait_cycles;
    begin
      wait_cycles = 0;
      #1;
      while (!output_0_valid) begin
        @(posedge clock);
        #1;
        wait_cycles = wait_cycles + 1;
        if (wait_cycles == 16)
          $fatal(1, "%s", message);
      end
      check(output_0_data == data && output_0_tag == 2'd2,
            "Temporal result carried the wrong data or configured tag");
      @(posedge clock);
      #1;
      check(!output_0_valid, "Consumed Temporal result remained valid");
    end
  endtask

  task automatic expect_silence(string message);
    begin
      repeat (8) begin
        @(posedge clock);
        #1;
        check(!output_0_valid, message);
      end
    end
  endtask

)sv";
  testbench << loom::hardware::test::portableAxiLiteDriverTasks();
  testbench << loom::hardware::test::portableCycleWatchdog();
  testbench << R"sv(

  initial begin
    clock = 0;
    reset = 1;
    input_0_data = 0;
    input_0_tag = 0;
    input_0_valid = 0;
    input_1_data = 0;
    input_1_tag = 0;
    input_1_valid = 0;
)sv";
  testbench << (sharedOutput ? "    output_0_ready_base = 0;\n"
                               "    refuse_enabled = 0;\n"
                               "    refused_tag = 0;\n"
                             : "    output_0_ready = 1;\n");
  if (gate)
    testbench << "    output_1_ready = 1;\n";
  testbench << loom::hardware::test::portableAxiLiteInitialization();
  testbench << R"sv(    repeat (2) @(posedge clock);
    #1 reset = 0;
    #1;
    check(!input_0_ready && !input_1_ready && !output_0_valid,
          "Disabled Temporal PE exchanged a token");

)sv";
  testbench << take(test, loom::hardware::test::portableAxiLiteProgramAndVerify(
                              artifact.target, artifact.image));
  if (gate)
    testbench << R"sv(    send_input_1(2'd1, 8'd9);
    expect_silence("Gate published without its phase head");
    send_input_0(2'd1, 8'd1);
    expect_gate(0, 8'd0, 1, 8'd9,
                "Gate FirstTrue case did not publish the value alone");
    send_input_1(2'd1, 8'd5);
    send_input_0(2'd1, 8'd1);
    expect_gate(1, 8'd1, 1, 8'd5,
                "Gate ContinueTrue case did not publish phase and value");
    send_input_1(2'd1, 8'd3);
    send_input_0(2'd1, 8'd0);
    expect_gate(1, 8'd0, 0, 8'd0,
                "Gate Close case did not publish the closing phase alone");
    send_input_1(2'd1, 8'd7);
    send_input_0(2'd1, 8'd1);
    expect_gate(0, 8'd0, 1, 8'd7,
                "Context state bank did not return to Closed after Close");
    expect_silence("Temporal PE published a token without a transition");
    $finish;
  end
endmodule
)sv";
  else if (sharedOutput)
    testbench << R"sv(    // The output port stays closed while the add (context 0) and the
    // multiply (context 1) both capture a result behind the one FU boundary
    // output. The port then refuses whichever result the FU presents, so the
    // other held result must retire first; releasing the refusal retires the
    // remaining one. Each handoff retires exactly one result.
    send_input_1(2'd1, 8'd4);
    send_input_0(2'd1, 8'd3);
    send_input_1(2'd3, 8'd6);
    send_input_0(2'd3, 8'd5);
    repeat (8) @(posedge clock);
    #1;
    check(output_0_valid && (output_0_tag == 2'd2 || output_0_tag == 2'd3),
          "Shared FU output presented no held result");
    refused_tag = output_0_tag;
    refuse_enabled = 1;
    @(negedge clock);
    output_0_ready_base = 1;
    if (refused_tag == 2'd2)
      expect_shared_result(2'd3, 8'd30,
                           "Refused sum held the shared FU output against the product");
    else
      expect_shared_result(2'd2, 8'd7,
                           "Refused product held the shared FU output against the sum");
    @(negedge clock);
    refuse_enabled = 0;
    if (refused_tag == 2'd2)
      expect_shared_result(2'd2, 8'd7, "Released sum did not retire");
    else
      expect_shared_result(2'd3, 8'd30, "Released product did not retire");
    expect_silence("Shared FU output published a third token");
    $finish;
  end
endmodule
)sv";
  else
    testbench << R"sv(    send_input_1(2'd1, 8'd9);
    expect_result(8'd9,
                  "Invariant Init case did not fire on its Init head alone");
    send_input_0(2'd1, 8'd1);
    expect_result(8'd9,
                  "Invariant Replay case did not fire on its phase head alone");
    send_input_0(2'd1, 8'd1);
    expect_result(8'd9, "Invariant Replay case lost its latched value");
    send_input_0(2'd1, 8'd0);
    expect_silence("Invariant Close case published a value");
    send_input_1(2'd1, 8'd5);
    expect_result(8'd5,
                  "Context state bank did not return to Initial after Close");
    expect_silence("Temporal PE published a token without a transition");
    $finish;
  end
endmodule
)sv";
  std::ofstream(root / (prefix + ".ys")) << "\nread_verilog -sv " << prefix
                                         << R"ys(_module.sv
hierarchy -check -top loom_module
check -assert
proc
synth -top loom_module
check -assert
select -assert-none loom_module/t:$dlatch loom_module/t:$_DLATCH_*
)ys";
}

} // namespace

int main(int argc, char **argv) {
  require("main", argc == 2, "expected exactly one output directory");
  const std::filesystem::path root(argv[1]);
  std::filesystem::create_directories(root);
  for (const Fixture fixture :
       {Fixture::Invariant, Fixture::Gate, Fixture::SharedOutput})
    writeArtifacts(root, buildArtifact(root / "store" / fixtureName(fixture).str(),
                                       fixture));
  return 0;
}
