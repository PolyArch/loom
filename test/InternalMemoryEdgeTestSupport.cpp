#include "InternalMemoryEdgeTestSupport.h"

#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/MemoryActorContractDomain.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/IR/MemoryConnectivityContract.h"
#include "Fabric/IR/MemoryOperationPort.h"
#include "Fabric/IR/OperationResourceContract.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::test {
namespace {

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    llvm::report_fatal_error(llvm::Twine(llvm::toString(value.takeError())));
  return std::move(*value);
}

::fabric::UnsignedDomain singleton(std::uint64_t value) {
  return take(::fabric::UnsignedDomain::fromCanonical({{value, value}}));
}

::fabric::ResourceContract memoryPortResourceContract() {
  ::fabric::ResourceContractDeclaration declaration;
  declaration.states = {::fabric::ResourceStateDeclaration{
      ::fabric::StateKey(0),
      {{::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1),
        ::fabric::CapacityUnits(0)}}}};
  declaration.requesters = {::fabric::RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 2;
  declaration.timingContracts = {{::fabric::TimingContractKey(0), {0, 1}}};
  declaration.usePatterns = {
      {::fabric::UsePatternKey(0),
       ::fabric::RequesterKey(0),
       ::fabric::EligibilityKey(0),
       ::fabric::EventKey(0),
       ::fabric::EventKey(1),
       std::nullopt,
       ::fabric::TimingContractKey(0),
       {{::fabric::ClaimKey(0), ::fabric::StateKey(0),
         ::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1)}},
       {{{::fabric::ClaimKey(0)}}}}};
  return take(::fabric::ResourceContract::create(std::move(declaration)));
}

::fabric::MemoryOperationPortDeclaration memoryPort(bool reads) {
  auto alignment = take(::fabric::AlignmentDomain::create(
      take(::fabric::UnsignedDomain::fromCanonical({{0, 63}}))));
  auto read = take(
      ::fabric::ClosedEnumDomain<::fabric::ReadSubwordSemantics>::fromCanonical(
          {reads ? ::fabric::ReadSubwordSemantics::ZeroExtend
                 : ::fabric::ReadSubwordSemantics::NotApplicable}));
  auto write = take(
      ::fabric::ClosedEnumDomain<
          ::fabric::WriteSubwordSemantics>::fromCanonical(
          {reads ? ::fabric::WriteSubwordSemantics::NotApplicable
                 : ::fabric::WriteSubwordSemantics::ByteEnable}));
  auto address =
      take(::fabric::MemoryAddressDomain::rootRelative(singleton(64)));
  auto access = take(::fabric::MemoryAccessClass::create(
      ::dataflow::semantics::MemoryAccessForm::Element, singleton(32),
      singleton(1),
      {{::dataflow::semantics::MemoryMaskForm::Absent,
        ::fabric::InactiveLaneSemantics::NotApplicable},
       {::dataflow::semantics::MemoryMaskForm::Dynamic,
        reads ? ::fabric::InactiveLaneSemantics::SuppressAndZeroFill
              : ::fabric::InactiveLaneSemantics::Suppress}},
      std::move(alignment), std::move(read), std::move(write),
      std::move(address)));
  auto accessDomain = take(
      ::fabric::ParameterizedMemoryAccessDomain::create({std::move(access)}));
  ::fabric::MemoryActorContractClause plain =
      ::fabric::LoadStorePlainContractClause{{false}};
  auto actorDomain = take(::fabric::MemoryActorContractDomain::create(
      reads ? ::dataflow::OperationSchemaId::DataflowLoad
            : ::dataflow::OperationSchemaId::DataflowStore,
      {plain}));

  using Role = ::dataflow::semantics::ServiceValueRole;
  ::fabric::MemoryCapabilityAlternativeRecord alternative{
      std::move(actorDomain),
      reads ? std::vector<::fabric::MemoryRoleEndpointBindingRecord>{
                  {Role::Address, 0}, {Role::Data, 7}, {Role::Mask, 1},
                  {Role::Control, 2}, {Role::Completion, 8}}
            : std::vector<::fabric::MemoryRoleEndpointBindingRecord>{
                  {Role::Address, 3}, {Role::Data, 4}, {Role::Mask, 5},
                  {Role::Control, 6}, {Role::Completion, 9}},
      std::move(accessDomain),
      {::fabric::UsePatternKey(0)}};
  return {reads ? std::vector<std::uint64_t>{0, 1, 2, 7, 8}
                : std::vector<std::uint64_t>{3, 4, 5, 6, 9},
          memoryPortResourceContract(),
          {{::fabric::MemoryPortTransactionProjection::Direct}},
          {std::move(alternative)}};
}

} // namespace

fabric::FinalizedFabricRoot
buildInternalMemoryEdgeFabric(ArtifactStore &store,
                              ::fabric::Schedule schedule) {
  using adg::MemoryConnectivitySpec;
  using adg::MemoryEngineSpec;
  using adg::MemorySpec;
  using adg::PortType;

  const auto bits8 = take(PortType::bits(8));
  const auto bits128 = take(PortType::bits(128));
  const auto manager =
      take(PortType::memory({PortType::kDynamicExtent}, bits8));
  std::vector<PortType> memoryInputs{manager};
  for (std::uint32_t width : {64u, 4u, 0u, 64u, 128u, 4u, 0u})
    memoryInputs.push_back(take(schedule == ::fabric::Schedule::Temporal
                                    ? PortType::taggedBits(width, 4)
                                    : PortType::bits(width)));
  std::vector<PortType> memoryOutputs;
  for (std::uint32_t width : {128u, 0u, 0u})
    memoryOutputs.push_back(take(schedule == ::fabric::Schedule::Temporal
                                     ? PortType::taggedBits(width, 4)
                                     : PortType::bits(width)));

  ::fabric::MemoryDispatchTarget managerTarget(
      std::in_place_type<::fabric::ManagerMemoryDispatchTarget>,
      ::fabric::ManagerMemoryDispatchTarget{0});
  ::fabric::MemoryConnectivityDeclaration connectivity;
  ::fabric::MemoryOperationPortDispatchDeclaration readDispatch;
  readDispatch.capabilityTargetDomains = {{managerTarget}};
  ::fabric::MemoryOperationPortDispatchDeclaration writeDispatch;
  writeDispatch.capabilityTargetDomains = {{managerTarget}};
  connectivity.operationPorts = {std::move(readDispatch),
                                 std::move(writeDispatch)};
  connectivity.internalConnections = {{8, 6}};
  auto engine = schedule == ::fabric::Schedule::Temporal
                    ? MemoryEngineSpec::temporal(
                          8, {memoryPort(true), memoryPort(false)})
                    : MemoryEngineSpec::spatial(
                          {memoryPort(true), memoryPort(false)});
  auto spec = take(MemorySpec::create(
      memoryInputs, memoryOutputs, {0}, {}, std::move(engine), std::nullopt,
      take(MemoryConnectivitySpec::create(std::move(connectivity)))));

  std::vector<PortType> moduleInputs = memoryInputs;
  std::vector<PortType> moduleOutputs = memoryOutputs;
  moduleInputs.push_back(bits128);
  moduleOutputs.push_back(bits128);
  adg::DesignBuilder builder(store);
  auto spatial = take(builder.createSpatialCore(
      "memory-internal-edge", moduleInputs, moduleOutputs));
  std::vector<adg::SpatialValue> inputValues;
  for (std::size_t ordinal = 0; ordinal < memoryInputs.size(); ++ordinal)
    inputValues.push_back(take(spatial.input(ordinal)));
  auto outputs = take(spatial.addMemory(inputValues, spec));
  std::vector<adg::SpatialValue> results(outputs.values().begin(),
                                         outputs.values().end());
  results.push_back(take(spatial.input(memoryInputs.size())));
  if (llvm::Error error = spatial.close(results))
    llvm::report_fatal_error(
        llvm::Twine(llvm::toString(std::move(error))));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    llvm::report_fatal_error(llvm::Twine(
        "internal-edge Fabric did not publish exactly one root"));
  return design.roots().front();
}

} // namespace loom::test
