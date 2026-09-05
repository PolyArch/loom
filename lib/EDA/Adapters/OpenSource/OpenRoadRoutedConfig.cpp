#include "EDA/Adapters/OpenSource/OpenRoadRouted.h"

#include "OpenRoadConfiguration.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace loom::eda::open_source {
namespace {
constexpr llvm::StringLiteral kSchema = "loom.openroad_routed_config";
constexpr llvm::StringLiteral kVersion = "1.0";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "openroad_routed_config_invalid: " + message);
}

llvm::Expected<OpenRoadRoutingParameters>
parseRouting(const llvm::json::Object &object) {
  if (object.size() != 3)
    return invalid("routing requires exactly three policy fields");
  const auto minimum = object.getString("minimum_routing_layer");
  const auto maximum = object.getString("maximum_routing_layer");
  const auto *cutoff = object.get("via_access_cutoff_layer");
  if (!minimum || !maximum || !cutoff ||
      (!cutoff->getAsNull() && !cutoff->getAsString()))
    return invalid("routing policy has a missing or mistyped layer field");
  OpenRoadRoutingParameters parameters{minimum->str(), maximum->str(),
                                       std::nullopt};
  if (const auto name = cutoff->getAsString())
    parameters.viaAccessCutoffLayer = name->str();
  if (llvm::Error error = validateOpenRoadRoutingParameters(parameters))
    return std::move(error);
  return parameters;
}

void writeRouting(llvm::json::OStream &json,
                  const OpenRoadRoutingParameters &parameters) {
  json.object([&] {
    json.attribute("minimum_routing_layer", parameters.minimumRoutingLayer);
    json.attribute("maximum_routing_layer", parameters.maximumRoutingLayer);
    if (parameters.viaAccessCutoffLayer)
      json.attribute("via_access_cutoff_layer",
                     *parameters.viaAccessCutoffLayer);
    else
      json.attribute("via_access_cutoff_layer", nullptr);
  });
}
} // namespace

llvm::Error
validateOpenRoadRoutingParameters(const OpenRoadRoutingParameters &parameters) {
  if (!detail::isPortableIdentifier(parameters.minimumRoutingLayer) ||
      !detail::isPortableIdentifier(parameters.maximumRoutingLayer) ||
      (parameters.viaAccessCutoffLayer &&
       !detail::isPortableIdentifier(*parameters.viaAccessCutoffLayer)))
    return invalid("routing layer names must be portable identifiers");
  return llvm::Error::success();
}

llvm::Expected<OpenRoadRoutingParameters>
parseOpenRoadRoutingParametersJson(llvm::StringRef json) {
  auto parsed = llvm::json::parse(json);
  if (!parsed)
    return invalid(llvm::toString(parsed.takeError()));
  const auto *object = parsed->getAsObject();
  if (!object)
    return invalid("routing policy must be an object");
  return parseRouting(*object);
}

llvm::ArrayRef<std::uint8_t> openRoadRoutedConfigSchemaDescriptorBytes() {
  static const std::string descriptor = (kSchema + ":" + kVersion).str();
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

llvm::Expected<std::vector<std::uint8_t>>
encodeOpenRoadRoutedConfig(const OpenRoadRoutedConfig &config) {
  if (llvm::Error error = validateOpenRoadRoutingParameters(config.routing))
    return std::move(error);
  auto physical = encodeOpenRoadPlacedConfig(config.physical);
  if (!physical)
    return physical.takeError();
  llvm::SmallString<1024> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", kSchema);
    json.attribute("version", kVersion);
    json.attributeBegin("physical");
    json.rawValue(llvm::StringRef(
        reinterpret_cast<const char *>(physical->data()), physical->size()));
    json.attributeEnd();
    json.attributeBegin("routing");
    writeRouting(json, config.routing);
    json.attributeEnd();
  });
  return std::vector<std::uint8_t>(storage.begin(), storage.end());
}

llvm::Expected<OpenRoadRoutedConfig>
decodeOpenRoadRoutedConfig(llvm::ArrayRef<std::uint8_t> bytes) {
  const llvm::StringRef contents(reinterpret_cast<const char *>(bytes.data()),
                                 bytes.size());
  auto parsed = llvm::json::parse(contents);
  if (!parsed)
    return invalid(llvm::toString(parsed.takeError()));
  const auto *object = parsed->getAsObject();
  if (!object || object->size() != 4 ||
      object->getString("schema") != kSchema ||
      object->getString("version") != kVersion)
    return invalid("routed configuration schema or fields are invalid");
  const auto *physicalObject = object->getObject("physical");
  const auto *routingObject = object->getObject("routing");
  if (!physicalObject || !routingObject)
    return invalid(
        "routed configuration requires physical and routing objects");
  auto physical = detail::parseOpenRoadPlacedConfigObject(*physicalObject);
  if (!physical)
    return physical.takeError();
  auto routing = parseRouting(*routingObject);
  if (!routing)
    return routing.takeError();
  OpenRoadRoutedConfig config{std::move(*physical), std::move(*routing)};
  auto canonical = encodeOpenRoadRoutedConfig(config);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return invalid("routed configuration is not canonical");
  return config;
}

llvm::Error validateCanonicalOpenRoadRoutedConfig(
    llvm::ArrayRef<std::uint8_t> bytes,
    const ComponentViewDigest &suppliedDigest) {
  auto config = decodeOpenRoadRoutedConfig(bytes);
  if (!config)
    return config.takeError();
  return validateComponentViewDigest(
      openRoadRoutedConfigSchemaDescriptorBytes(), bytes, suppliedDigest);
}
} // namespace loom::eda::open_source
