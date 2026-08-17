#include "TemporalMappingFabricTestSupport.h"

#include "ADG/FuLibrary.h"
#include "Fabric/IR/OperationResourceContract.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "temporal mapping Fabric test support: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

} // namespace

void loom::test::addTokenSyncFu(adg::PeBuilder &pe,
                                llvm::ArrayRef<adg::PeValue> inputs,
                                const adg::PortType &type,
                                const ::fabric::ResourceContract &contract) {
  using adg::FuCapabilityTemplateSpec;
  using adg::FuSpec;
  using adg::OperationCapabilitySpec;

  const std::vector<adg::PortType> types(4, type);
  auto fu = take(pe.addFu(inputs, FuSpec{types, types}));
  std::vector<adg::FuValue> fuInputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    fuInputs.push_back(take(fu.input(ordinal)));
  auto operation = take(fu.addOperation(
      fuInputs,
      OperationCapabilitySpec{::fabric::ImplementationFamilyId::TokenSync,
                              ::fabric::RoutedTokenParams{128, 4},
                              {::dataflow::OperationSchemaId::DataflowSync},
                              types,
                              contract}));
  requireSuccess(
      fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{operation}, {}}));
  std::vector<adg::FuValue> outputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    outputs.push_back(take(operation.output(ordinal)));
  requireSuccess(fu.close(outputs));
}

loom::fabric::FinalizedFabricRoot
loom::test::buildBoundaryTemporalFabric(ArtifactStore &store) {
  using namespace loom::adg;

  const PortType bits4 = take(PortType::bits(4));
  const PortType bits128 = take(PortType::bits(128));
  const PortType tagged128 = take(PortType::taggedBits(128, 4));
  std::vector<PortType> moduleInputs;
  moduleInputs.reserve(10);
  for (unsigned input = 0; input != 5; ++input) {
    moduleInputs.push_back(bits128);
    moduleInputs.push_back(bits4);
  }
  std::vector<PortType> moduleOutputs;
  moduleOutputs.reserve(8);
  for (unsigned output = 0; output != 4; ++output) {
    moduleOutputs.push_back(bits128);
    moduleOutputs.push_back(bits4);
  }

  DesignBuilder builder(store);
  auto spatial = take(builder.createSpatialCore("boundary-temporal",
                                                moduleInputs, moduleOutputs));
  std::vector<SpatialValue> taggedInputs;
  taggedInputs.reserve(5);
  for (unsigned input = 0; input != 5; ++input) {
    const std::array<SpatialValue, 2> boundaryInputs = {
        take(spatial.input(input * 2)), take(spatial.input(input * 2 + 1))};
    auto outputs = take(spatial.addBoundary(
        boundaryInputs, BoundarySpec::s2t(bits128, bits4, tagged128)));
    taggedInputs.push_back(outputs.front());
  }
  auto pe = take(spatial.addPe(
      taggedInputs,
      PeSpec::temporal(
          std::vector<PortType>(5, bits128),
          std::vector<PortType>(4, tagged128),
          TemporalPeParameters{2, FuConfigurationMode::PerInstruction,
                               ::fabric::OperandBufferMode::PerInstruction, 2,
                               std::nullopt})));
  std::vector<PeValue> peInputs;
  peInputs.reserve(4);
  for (unsigned input = 0; input != 4; ++input)
    peInputs.push_back(take(pe.input(input)));
  addTokenSyncFu(pe, peInputs, bits128,
                 ::fabric::oneCycleElasticOperationResourceContract());
  requireSuccess(pe.close());

  std::vector<SpatialValue> untaggedOutputs;
  untaggedOutputs.reserve(8);
  for (unsigned output = 0; output != 4; ++output) {
    auto split = take(
        spatial.addBoundary({take(pe.output(output))},
                            BoundarySpec::t2s(tagged128, {bits128, bits4})));
    untaggedOutputs.insert(untaggedOutputs.end(), split.values().begin(),
                           split.values().end());
  }
  requireSuccess(spatial.close(untaggedOutputs));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("boundary Temporal Fabric did not publish exactly one root");
  return design.roots().front();
}
