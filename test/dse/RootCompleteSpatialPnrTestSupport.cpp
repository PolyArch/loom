#include "RootCompleteSpatialPnrTestSupport.h"

#include "ADG/Builder.h"
#include "ADG/FuLibrary.h"
#include "Common/ArtifactStore.h"
#include "Fabric/IR/OperationResourceContract.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <utility>
#include <vector>

namespace loom::test {
namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "root-complete Spatial PnR fixture failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
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

fabric::FinalizedFabricRoot buildSpatialCore(ArtifactStore &store,
                                             std::uint32_t payloadWidth) {
  const adg::PortType payloadType = take(adg::PortType::bits(payloadWidth));
  const std::vector<adg::PortType> types(4, payloadType);
  adg::DesignBuilder builder(store);
  auto spatial = take(builder.createSpatialCore("sync", types, types));
  std::vector<adg::SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    spatialInputs.push_back(take(spatial.input(ordinal)));
  auto pe =
      take(spatial.addPe(spatialInputs, adg::PeSpec::spatial(types, types)));
  std::vector<adg::PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    peInputs.push_back(take(pe.input(ordinal)));
  auto fu = take(pe.addFu(peInputs, adg::FuSpec{types, types}));
  std::vector<adg::FuValue> fuInputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    fuInputs.push_back(take(fu.input(ordinal)));
  auto operation = take(fu.addOperation(
      fuInputs, adg::OperationCapabilitySpec{
                    ::fabric::ImplementationFamilyId::TokenSync,
                    ::fabric::RoutedTokenParams{payloadWidth, 4},
                    {::dataflow::OperationSchemaId::DataflowSync},
                    types,
                    ::fabric::oneCycleElasticOperationResourceContract()}));
  requireSuccess(
      fu.addCapabilityTemplate(adg::FuCapabilityTemplateSpec{{operation}, {}}));
  std::vector<adg::FuValue> fuOutputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    fuOutputs.push_back(take(operation.output(ordinal)));
  requireSuccess(fu.close(fuOutputs));
  requireSuccess(pe.close());
  std::vector<adg::SpatialValue> outputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    outputs.push_back(take(pe.output(ordinal)));
  requireSuccess(spatial.close(outputs));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("SpatialCore fixture did not publish exactly one Fabric root");
  return design.roots().front();
}

fabric::FinalizedFabricRoot
buildLineageSpatialCore(ArtifactStore &store, std::uint32_t payloadWidth) {
  const adg::PortType payloadType = take(adg::PortType::bits(payloadWidth));
  const std::vector<adg::PortType> types(4, payloadType);
  const std::vector<adg::PortType> tokenInputTypes(5, payloadType);
  adg::DesignBuilder builder(store);
  auto spatial = take(builder.createSpatialCore("lineage-sync", types, types));
  auto network = take(
      spatial.addMeshSwitchNetwork(take(adg::MeshSwitchNetworkSpec::spatial(
          2, 2, 2, payloadType,
          {{0, 0, {payloadType, payloadType}, {payloadType, payloadType}},
           {0, 1, {payloadType, payloadType}, {payloadType, payloadType}},
           {1, 0, types, types},
           {1, 1, tokenInputTypes, types}}))));

  auto upperBoundary = take(network.attachment(0));
  auto lowerBoundary = take(network.attachment(1));
  auto vectorCompute = take(network.attachment(2));
  auto tokenControl = take(network.attachment(3));
  requireSuccess(upperBoundary.connectOutputs(
      {take(spatial.input(0)), take(spatial.input(1))}));
  requireSuccess(lowerBoundary.connectOutputs(
      {take(spatial.input(2)), take(spatial.input(3))}));

  auto vectorPe = take(spatial.addPe(vectorCompute.inputs(),
                                     adg::PeSpec::spatial(types, types)));
  std::vector<adg::PeValue> vectorInputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    vectorInputs.push_back(take(vectorPe.input(ordinal)));
  requireSuccess(adg::addVectorComputeFu(vectorPe, vectorInputs,
                                         {payloadWidth, payloadWidth}));
  const ::fabric::IntegerWidthSet integerWidths =
      ::fabric::IntegerWidthSet::get(
          {::fabric::IntegerWidth::I8, ::fabric::IntegerWidth::I16,
           ::fabric::IntegerWidth::I32, ::fabric::IntegerWidth::I64});
  const ::fabric::FloatFormatSet floatFormats = ::fabric::FloatFormatSet::get(
      {::fabric::FloatFormat::F16, ::fabric::FloatFormat::BF16,
       ::fabric::FloatFormat::F32, ::fabric::FloatFormat::F64});
  const adg::VectorStructuralFuParameters structural{
      payloadWidth, payloadWidth, 64,
      ::fabric::FixedVectorSliceAlignMergeParams{
          integerWidths, floatFormats, payloadWidth, payloadWidth, 0,
          ::fabric::ResolvedIndexWidthSet::get({})},
      ::fabric::FixedVectorShuffleParams{integerWidths, floatFormats,
                                         payloadWidth, payloadWidth, 32, 8, 4}};
  requireSuccess(adg::addVectorStructuralFu(
      vectorPe, llvm::ArrayRef<adg::PeValue>(vectorInputs).take_front(2),
      structural));
  requireSuccess(vectorPe.close());
  std::vector<adg::SpatialValue> vectorOutputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    vectorOutputs.push_back(take(vectorPe.output(ordinal)));
  requireSuccess(vectorCompute.connectOutputs(vectorOutputs));

  auto tokenPe = take(spatial.addPe(
      tokenControl.inputs(), adg::PeSpec::spatial(tokenInputTypes, types)));
  std::vector<adg::PeValue> tokenInputs;
  for (std::size_t ordinal = 0; ordinal != tokenInputTypes.size(); ++ordinal)
    tokenInputs.push_back(take(tokenPe.input(ordinal)));
  requireSuccess(adg::addTokenControlFu(
      tokenPe, tokenInputs, {payloadWidth, std::min(payloadWidth, 64U)}));
  requireSuccess(tokenPe.close());
  std::vector<adg::SpatialValue> tokenOutputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    tokenOutputs.push_back(take(tokenPe.output(ordinal)));
  requireSuccess(tokenControl.connectOutputs(tokenOutputs));

  std::vector<adg::SpatialValue> outputs(upperBoundary.inputs().begin(),
                                         upperBoundary.inputs().end());
  outputs.insert(outputs.end(), lowerBoundary.inputs().begin(),
                 lowerBoundary.inputs().end());
  requireSuccess(spatial.close(outputs));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("lineage SpatialCore fixture did not publish exactly one Fabric root");
  auto root = design.roots().front();
  if (root.view().fifoOccurrences().size() != 16 ||
      root.view().switchOccurrences().size() < 12)
    fail("lineage SpatialCore lost its finite multi-hop mesh topology");
  return root;
}

} // namespace loom::test
