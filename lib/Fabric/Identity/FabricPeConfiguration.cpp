#include "Fabric/Identity/FabricPeConfiguration.h"

#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace loom::fabric {
namespace {

llvm::Error rejected(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_pe_configuration_rejected: " +
                                     message);
}

bool containsEndpoint(llvm::ArrayRef<FabricTransportEndpointRef> endpoints,
                      const FabricTransportEndpointRef &endpoint) {
  return llvm::is_contained(endpoints, endpoint);
}

CanonicalSemanticBytes encodeTag(std::uint32_t tag) {
  FabricByteWriter writer;
  writer.tag(tag);
  return CanonicalSemanticBytes(writer.take());
}

template <typename Ref>
CanonicalSemanticBytes encodeTaggedRef(std::uint32_t tag,
                                       const Ref &reference) {
  FabricByteWriter writer;
  writer.tag(tag);
  encodeFabricRef(writer, reference);
  return CanonicalSemanticBytes(writer.take());
}

} // namespace

const FabricPeConfigurationFieldView *
FabricSpatialPeConfigurationSchemaView::find(
    const FabricSemanticConfigFieldRef &field) const {
  const auto found =
      llvm::find_if(fields_, [&](const FabricPeConfigurationFieldView &entry) {
        return entry.reference == field;
      });
  return found == fields_.end() ? nullptr : &*found;
}

llvm::Expected<std::vector<FabricPeConfigurationValue>>
FabricSpatialPeConfigurationSchemaView::finiteDomain(
    const FabricSemanticConfigFieldRef &field) const {
  const FabricPeConfigurationFieldView *descriptor = find(field);
  if (!descriptor)
    return rejected("field is absent from the Spatial PE schema");

  std::vector<FabricPeConfigurationValue> domain;
  switch (descriptor->kind) {
  case FabricPeConfigurationFieldKind::Activation:
    domain.reserve(1 + fuOccurrences_.size());
    domain.emplace_back(FabricPeDisabled{});
    for (FabricFuOccurrenceRef fu : fuOccurrences_)
      domain.emplace_back(FabricPeActive{fu});
    return domain;
  case FabricPeConfigurationFieldKind::InputSelector:
    domain.reserve(1 + 2 * inputEndpoints_.size());
    domain.emplace_back(FabricPeDisconnected{});
    for (const FabricTransportEndpointRef &endpoint : inputEndpoints_)
      domain.emplace_back(FabricPeRoute{endpoint});
    for (const FabricTransportEndpointRef &endpoint : inputEndpoints_)
      domain.emplace_back(FabricPeInputDiscard{endpoint});
    return domain;
  case FabricPeConfigurationFieldKind::OutputSelector:
    domain.reserve(2 + outputEndpoints_.size());
    domain.emplace_back(FabricPeDisconnected{});
    for (const FabricTransportEndpointRef &endpoint : outputEndpoints_)
      domain.emplace_back(FabricPeRoute{endpoint});
    domain.emplace_back(FabricPeOutputDiscard{});
    return domain;
  }
  return rejected("field has an unknown role");
}

llvm::Expected<CanonicalSemanticBytes>
FabricSpatialPeConfigurationSchemaView::encode(
    const FabricSemanticConfigFieldRef &field,
    const FabricPeConfigurationValue &value) const {
  const FabricPeConfigurationFieldView *descriptor = find(field);
  if (!descriptor)
    return rejected("field is absent from the Spatial PE schema");

  switch (descriptor->kind) {
  case FabricPeConfigurationFieldKind::Activation:
    if (std::holds_alternative<FabricPeDisabled>(value))
      return encodeTag(0);
    if (const auto *active = std::get_if<FabricPeActive>(&value)) {
      if (!llvm::is_contained(fuOccurrences_, active->fu))
        return rejected("activation selects a foreign FU occurrence");
      return encodeTaggedRef(1, active->fu);
    }
    return rejected("activation field received a selector value");
  case FabricPeConfigurationFieldKind::InputSelector:
    if (std::holds_alternative<FabricPeDisconnected>(value))
      return encodeTag(0);
    if (const auto *route = std::get_if<FabricPeRoute>(&value)) {
      if (!containsEndpoint(inputEndpoints_, route->endpoint))
        return rejected("input Route selects a foreign or wrong-role endpoint");
      return encodeTaggedRef(1, route->endpoint);
    }
    if (const auto *discard = std::get_if<FabricPeInputDiscard>(&value)) {
      if (!containsEndpoint(inputEndpoints_, discard->endpoint))
        return rejected(
            "input Discard selects a foreign or wrong-role endpoint");
      return encodeTaggedRef(2, discard->endpoint);
    }
    return rejected("input selector received a value of the wrong kind");
  case FabricPeConfigurationFieldKind::OutputSelector:
    if (std::holds_alternative<FabricPeDisconnected>(value))
      return encodeTag(0);
    if (const auto *route = std::get_if<FabricPeRoute>(&value)) {
      if (!containsEndpoint(outputEndpoints_, route->endpoint))
        return rejected(
            "output Route selects a foreign or wrong-role endpoint");
      return encodeTaggedRef(1, route->endpoint);
    }
    if (std::holds_alternative<FabricPeOutputDiscard>(value))
      return encodeTag(2);
    return rejected("output selector received a value of the wrong kind");
  }
  return rejected("field has an unknown role");
}

llvm::Expected<FabricPeConfigurationValue>
FabricSpatialPeConfigurationSchemaView::decode(
    const FabricSemanticConfigFieldRef &field,
    llvm::ArrayRef<std::uint8_t> bytes) const {
  const FabricPeConfigurationFieldView *descriptor = find(field);
  if (!descriptor)
    return rejected("field is absent from the Spatial PE schema");

  FabricByteReader reader(bytes);
  const std::uint32_t bound =
      descriptor->kind == FabricPeConfigurationFieldKind::Activation ? 2 : 3;
  auto tag = readFabricClosedTag(reader, bound, "PE configuration value");
  if (!tag)
    return tag.takeError();

  FabricPeConfigurationValue value;
  if (descriptor->kind == FabricPeConfigurationFieldKind::Activation) {
    if (*tag == 0) {
      value = FabricPeDisabled{};
    } else {
      FabricFuOccurrenceRef fu;
      if (llvm::Error error = decodeFabricRefInto(reader, fu))
        return error;
      value = FabricPeActive{fu};
    }
  } else if (descriptor->kind ==
             FabricPeConfigurationFieldKind::InputSelector) {
    if (*tag == 0) {
      value = FabricPeDisconnected{};
    } else {
      FabricTransportEndpointRef endpoint;
      if (llvm::Error error = decodeFabricRefInto(reader, endpoint))
        return error;
      value = *tag == 1
                  ? FabricPeConfigurationValue(FabricPeRoute{endpoint})
                  : FabricPeConfigurationValue(FabricPeInputDiscard{endpoint});
    }
  } else {
    if (*tag == 0) {
      value = FabricPeDisconnected{};
    } else if (*tag == 1) {
      FabricTransportEndpointRef endpoint;
      if (llvm::Error error = decodeFabricRefInto(reader, endpoint))
        return error;
      value = FabricPeRoute{endpoint};
    } else {
      value = FabricPeOutputDiscard{};
    }
  }

  if (!reader.empty())
    return rejected("value has trailing canonical bytes");
  auto encoded = encode(field, value);
  if (!encoded)
    return encoded.takeError();
  if (!encoded->bytes().equals(bytes))
    return rejected("value does not re-encode canonically");
  return value;
}

llvm::Expected<FabricSpatialPeConfigurationSchemaView>
FabricArtifactView::spatialPeConfigurationSchema(
    FabricPeOccurrenceRef occurrence) const {
  if (llvm::Error error = validateFabricRef(*this, occurrence))
    return error;
  if (peSchedule(occurrence) != ::fabric::Schedule::Spatial)
    return rejected("configuration schema requires a Spatial PE");

  std::vector<FabricFuOccurrenceRef> fus;
  for (FabricFuOccurrenceRef fu : fuOccurrences())
    if (parentPeOf(fu) == occurrence)
      fus.push_back(fu);

  const FabricConfigurationOwnerRef owner(
      FabricInventoryOwnerRef::of(occurrence));
  std::vector<FabricPeConfigurationFieldView> fields;
  fields.push_back({FabricSemanticConfigFieldRef{owner, 0},
                    FabricPeConfigurationFieldKind::Activation, std::nullopt});
  FabricOrdinal ordinal = 1;
  for (FabricFuOccurrenceRef fu : fus) {
    const auto fuOwner = FabricInventoryOwnerRef::of(fu);
    const std::uint64_t count =
        inventorySize(fuOwner, FabricInventoryKind::InputPort);
    for (FabricOrdinal port = 0; port < count; ++port)
      fields.push_back(
          {FabricSemanticConfigFieldRef{owner, ordinal++},
           FabricPeConfigurationFieldKind::InputSelector,
           FabricFuOccurrencePortRef{fu, FabricPortDirection::Input, port}});
  }
  for (FabricFuOccurrenceRef fu : fus) {
    const auto fuOwner = FabricInventoryOwnerRef::of(fu);
    const std::uint64_t count =
        inventorySize(fuOwner, FabricInventoryKind::OutputPort);
    for (FabricOrdinal port = 0; port < count; ++port)
      fields.push_back(
          {FabricSemanticConfigFieldRef{owner, ordinal++},
           FabricPeConfigurationFieldKind::OutputSelector,
           FabricFuOccurrencePortRef{fu, FabricPortDirection::Output, port}});
  }

  std::vector<FabricTransportEndpointRef> inputs;
  std::vector<FabricTransportEndpointRef> outputs;
  const FabricTransportEndpointOwnerRef endpointOwner =
      FabricTransportEndpointOwnerRef::of(occurrence);
  for (FabricOrdinal endpoint = 0;
       endpoint < transportEndpointCount(endpointOwner); ++endpoint) {
    const FabricTransportEndpointRef reference{endpointOwner, endpoint};
    const auto direction = transportEndpointDirection(reference);
    if (!direction)
      return rejected("PE transport inventory has no endpoint direction");
    (*direction == FabricPortDirection::Input ? inputs : outputs)
        .push_back(reference);
  }

  return FabricSpatialPeConfigurationSchemaView(
      occurrence, std::move(fields), std::move(fus), std::move(inputs),
      std::move(outputs));
}

} // namespace loom::fabric
