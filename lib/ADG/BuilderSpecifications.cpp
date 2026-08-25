#include "ADG/Builder.h"

#include "BuilderInternal.h"

#include <utility>

namespace loom::adg {

using detail::invalid;

llvm::Expected<PortType> PortType::bits(std::uint32_t width) {
  return PortType(Kind::Bits, width, 0, {});
}

llvm::Expected<PortType> PortType::taggedBits(std::uint32_t width,
                                              std::uint32_t tagWidth) {
  if (tagWidth == 0)
    return invalid("tagged Fabric port requires a positive tag width");
  return PortType(Kind::TaggedBits, width, tagWidth, {});
}

llvm::Expected<PortType> PortType::memory(llvm::ArrayRef<std::int64_t> shape,
                                          const PortType &elementType) {
  if (elementType.kind() == Kind::Memory)
    return invalid("Fabric memory element cannot itself be a memory port");
  if (elementType.kind() == Kind::TaggedBits)
    return invalid("Fabric memref element must be untagged bits");
  if (elementType.width() == 0)
    return invalid("Fabric memref element requires a positive data width");
  for (std::int64_t extent : shape)
    if (extent <= 0 && extent != PortType::kDynamicExtent)
      return invalid("Fabric memory shape contains an invalid extent");
  return PortType(Kind::Memory, elementType.width(), elementType.tagWidth(),
                  std::vector<std::int64_t>(shape.begin(), shape.end()));
}

PeSpec PeSpec::spatial(std::vector<PortType> inputTypes,
                       std::vector<PortType> outputTypes) {
  return PeSpec(::fabric::Schedule::Spatial, std::move(inputTypes),
                std::move(outputTypes), std::nullopt);
}

PeSpec PeSpec::temporal(std::vector<PortType> inputTypes,
                        std::vector<PortType> outputTypes,
                        TemporalPeParameters parameters) {
  return PeSpec(::fabric::Schedule::Temporal, std::move(inputTypes),
                std::move(outputTypes), std::move(parameters));
}

BoundarySpec BoundarySpec::s2t(const PortType &dataInput,
                               const PortType &tagInput,
                               const PortType &taggedOutput) {
  return {
      ::fabric::BoundaryDirection::S2t, {dataInput, tagInput}, {taggedOutput}};
}

BoundarySpec BoundarySpec::s2tWithConfiguredTag(const PortType &dataInput,
                                                const PortType &taggedOutput) {
  return {::fabric::BoundaryDirection::S2t, {dataInput}, {taggedOutput}};
}

BoundarySpec BoundarySpec::t2s(const PortType &taggedInput,
                               llvm::ArrayRef<PortType> outputs) {
  return {::fabric::BoundaryDirection::T2s,
          {taggedInput},
          std::vector<PortType>(outputs.begin(), outputs.end())};
}

SwitchSpec
SwitchSpec::spatial(std::vector<PortType> inputTypes,
                    std::vector<PortType> outputTypes,
                    std::vector<std::vector<std::uint32_t>> sourcesByOutput) {
  return {::fabric::Schedule::Spatial,
          std::move(inputTypes),
          std::move(outputTypes),
          std::move(sourcesByOutput),
          std::nullopt,
          std::nullopt};
}

SwitchSpec SwitchSpec::temporal(
    std::vector<PortType> inputTypes, std::vector<PortType> outputTypes,
    std::vector<std::vector<std::uint32_t>> sourcesByOutput,
    std::uint32_t routeTableSize,
    std::optional<::fabric::TemporalSwitchGrantPolicy> grantPolicy) {
  return {::fabric::Schedule::Temporal,
          std::move(inputTypes),
          std::move(outputTypes),
          std::move(sourcesByOutput),
          routeTableSize,
          std::move(grantPolicy)};
}

} // namespace loom::adg
