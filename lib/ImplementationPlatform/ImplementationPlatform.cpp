#include "ImplementationPlatform/ImplementationPlatform.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactStore.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::platform {
namespace {

struct CanonicalPlatformData final {
  ImplementationTarget target;
  std::vector<TechnologyCorner> corners;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "implementation_platform_invalid: " +
                                     message);
}

bool isIdentifierCharacter(char character) {
  const unsigned char value = static_cast<unsigned char>(character);
  return std::isalnum(value) || character == '.' || character == '_' ||
         character == '-' || character == '+' || character == ':' ||
         character == '/';
}

llvm::Error validateIdentifier(llvm::StringRef value,
                               llvm::StringRef field) {
  if (value.empty())
    return invalid(field + " must be nonempty");
  const auto isAlphanumeric = [](char character) {
    return std::isalnum(static_cast<unsigned char>(character)) != 0;
  };
  if (!isAlphanumeric(value.front()) || !isAlphanumeric(value.back()) ||
      !llvm::all_of(value, isIdentifierCharacter))
    return invalid(field + " is not a canonical ASCII identifier");
  return llvm::Error::success();
}

llvm::Error validateTarget(const ImplementationTarget &target) {
  return std::visit(
      [](const auto &value) -> llvm::Error {
        using Target = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Target, AsicTarget>) {
          if (llvm::Error error = validateIdentifier(
                  value.technologyIdentity, "technology_identity"))
            return error;
          return validateIdentifier(value.releaseIdentity,
                                    "release_identity");
        } else {
          if (value.vendor != FpgaVendor::AmdXilinx &&
              value.vendor != FpgaVendor::IntelAltera)
            return invalid("FPGA vendor is unknown");
          return validateIdentifier(value.deviceOrderingCode,
                                    "device_ordering_code");
        }
      },
      target);
}

llvm::Expected<CanonicalPlatformData>
canonicalize(ImplementationPlatformDraft draft) {
  if (llvm::Error error = validateTarget(draft.target))
    return std::move(error);
  if (draft.technologyCornerKeys.empty())
    return invalid("technology corner catalog must be nonempty");
  for (const std::string &key : draft.technologyCornerKeys)
    if (llvm::Error error = validateIdentifier(key, "corner_key"))
      return std::move(error);
  llvm::sort(draft.technologyCornerKeys);
  if (std::adjacent_find(draft.technologyCornerKeys.begin(),
                         draft.technologyCornerKeys.end()) !=
      draft.technologyCornerKeys.end())
    return invalid("technology corner catalog contains a duplicate key");
  if (draft.technologyCornerKeys.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()))
    return invalid("technology corner catalog exceeds the ID namespace");

  std::vector<TechnologyCorner> corners;
  corners.reserve(draft.technologyCornerKeys.size());
  for (std::size_t index = 0; index < draft.technologyCornerKeys.size();
       ++index)
    corners.push_back(TechnologyCorner{
        TechnologyCornerId(static_cast<std::uint64_t>(index)),
        std::move(draft.technologyCornerKeys[index])});
  return CanonicalPlatformData{std::move(draft.target), std::move(corners)};
}

llvm::StringRef vendorSpelling(FpgaVendor vendor) {
  switch (vendor) {
  case FpgaVendor::AmdXilinx:
    return "amd_xilinx";
  case FpgaVendor::IntelAltera:
    return "intel_altera";
  }
  llvm_unreachable("validated FPGA vendor is closed");
}

std::string serialize(const ImplementationPlatform &platform) {
  llvm::SmallString<512> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attributeObject("target", [&] {
      std::visit(
          [&](const auto &target) {
            using Target = std::decay_t<decltype(target)>;
            if constexpr (std::is_same_v<Target, AsicTarget>) {
              json.attribute("kind", "asic");
              json.attribute("technology_identity",
                             target.technologyIdentity);
              json.attribute("release_identity", target.releaseIdentity);
            } else {
              json.attribute("kind", "fpga");
              json.attribute("vendor", vendorSpelling(target.vendor));
              json.attribute("device_ordering_code",
                             target.deviceOrderingCode);
            }
          },
          platform.target());
    });
    json.attributeArray("technology_corners", [&] {
      for (const TechnologyCorner &corner : platform.technologyCorners()) {
        json.object([&] {
          json.attribute("corner_id", corner.id.value());
          json.attribute("corner_key", corner.key);
        });
      }
    });
  });
  return output.str().str();
}

llvm::Error rejectUnknownFields(const llvm::json::Object &object,
                                llvm::StringRef context,
                                llvm::ArrayRef<llvm::StringRef> allowed) {
  for (const auto &[key, value] : object)
    if (!llvm::is_contained(allowed, llvm::StringRef(key)))
      return invalid(context + " contains unknown field '" +
                     llvm::StringRef(key) + "'");
  return llvm::Error::success();
}

llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef key,
                                              llvm::StringRef context) {
  std::optional<llvm::StringRef> value = object.getString(key);
  if (!value)
    return invalid(context + " requires string field '" + key + "'");
  return *value;
}

llvm::Expected<ImplementationPlatformDraft> parse(llvm::StringRef body) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(body);
  if (!parsed)
    return invalid(llvm::toString(parsed.takeError()));
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return invalid("root must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *root, "root", {"target", "technology_corners"}))
    return std::move(error);

  const llvm::json::Object *targetObject = root->getObject("target");
  if (!targetObject)
    return invalid("root requires object field 'target'");
  auto kind = requireString(*targetObject, "kind", "target");
  if (!kind)
    return kind.takeError();

  ImplementationTarget target;
  if (*kind == "asic") {
    if (llvm::Error error = rejectUnknownFields(
            *targetObject, "target",
            {"kind", "technology_identity", "release_identity"}))
      return std::move(error);
    auto technology =
        requireString(*targetObject, "technology_identity", "target");
    if (!technology)
      return technology.takeError();
    auto release = requireString(*targetObject, "release_identity", "target");
    if (!release)
      return release.takeError();
    target = AsicTarget{technology->str(), release->str()};
  } else if (*kind == "fpga") {
    if (llvm::Error error = rejectUnknownFields(
            *targetObject, "target",
            {"kind", "vendor", "device_ordering_code"}))
      return std::move(error);
    auto vendor = requireString(*targetObject, "vendor", "target");
    if (!vendor)
      return vendor.takeError();
    FpgaVendor parsedVendor;
    if (*vendor == "amd_xilinx")
      parsedVendor = FpgaVendor::AmdXilinx;
    else if (*vendor == "intel_altera")
      parsedVendor = FpgaVendor::IntelAltera;
    else
      return invalid("target has unknown FPGA vendor");
    auto orderingCode =
        requireString(*targetObject, "device_ordering_code", "target");
    if (!orderingCode)
      return orderingCode.takeError();
    target = FpgaTarget{parsedVendor, orderingCode->str()};
  } else {
    return invalid("target has unknown kind");
  }

  const llvm::json::Array *corners = root->getArray("technology_corners");
  if (!corners)
    return invalid("root requires array field 'technology_corners'");
  std::vector<std::string> keys;
  keys.reserve(corners->size());
  for (std::size_t index = 0; index < corners->size(); ++index) {
    const llvm::json::Object *corner = (*corners)[index].getAsObject();
    if (!corner)
      return invalid("technology corner must be an object");
    if (llvm::Error error = rejectUnknownFields(
            *corner, "technology corner", {"corner_id", "corner_key"}))
      return std::move(error);
    const llvm::json::Value *idValue = corner->get("corner_id");
    const std::optional<std::uint64_t> id =
        idValue ? idValue->getAsUINT64() : std::nullopt;
    if (!id || *id != index)
      return invalid("technology corner IDs are not dense canonical order");
    auto key = requireString(*corner, "corner_key", "technology corner");
    if (!key)
      return key.takeError();
    keys.push_back(key->str());
  }
  return ImplementationPlatformDraft{std::move(target), std::move(keys)};
}

CanonicalSemanticBytes canonicalBytes(const ImplementationPlatform &platform) {
  const std::string body = serialize(platform);
  return CanonicalSemanticBytes(
      std::vector<std::uint8_t>(body.begin(), body.end()));
}

} // namespace

const TechnologyCorner *
ImplementationPlatform::findTechnologyCorner(TechnologyCornerId id) const {
  if (id.value() >= technologyCorners_.size())
    return nullptr;
  const TechnologyCorner &corner =
      technologyCorners_[static_cast<std::size_t>(id.value())];
  return corner.id == id ? &corner : nullptr;
}

llvm::Expected<FinalizedImplementationPlatform>
finalizeImplementationPlatform(ImplementationPlatformDraft draft,
                               const ArtifactStore &store) {
  auto data = canonicalize(std::move(draft));
  if (!data)
    return data.takeError();
  ImplementationPlatform platform(std::move(data->target),
                                  std::move(data->corners));
  CanonicalSemanticBytes bytes = canonicalBytes(platform);
  auto identity = store.put(implementationPlatformSchema, bytes);
  if (!identity)
    return identity.takeError();
  ArtifactRootReference reference{implementationPlatformSchema.identity.str(),
                                  implementationPlatformSchema.version,
                                  std::move(*identity)};
  return importImplementationPlatform(reference, store);
}

llvm::Expected<FinalizedImplementationPlatform>
importImplementationPlatform(const ArtifactRootReference &reference,
                             const ArtifactStore &store) {
  if (reference.schemaIdentity != implementationPlatformSchema.identity ||
      reference.schemaVersion != implementationPlatformSchema.version)
    return invalid("reference requires loom.implementation_platform 1.0");
  auto stored = store.get(implementationPlatformSchema, reference.artifact);
  if (!stored)
    return stored.takeError();
  const llvm::ArrayRef<std::uint8_t> bytes = stored->bytes();
  const llvm::StringRef body(reinterpret_cast<const char *>(bytes.data()),
                             bytes.size());
  auto draft = parse(body);
  if (!draft)
    return draft.takeError();
  auto data = canonicalize(std::move(*draft));
  if (!data)
    return data.takeError();
  ImplementationPlatform platform(std::move(data->target),
                                  std::move(data->corners));
  CanonicalSemanticBytes canonical = canonicalBytes(platform);
  if (!canonical.bytes().equals(bytes))
    return invalid("stored root is not canonical");
  if (finalizeArtifactIdentity(implementationPlatformSchema, canonical) !=
      reference.artifact)
    return invalid("reference has a stale platform identity");
  return FinalizedImplementationPlatform(reference, std::move(canonical),
                                         std::move(platform));
}

llvm::Expected<TechnologyCorner>
resolveTechnologyCorner(const TechnologyCornerRef &reference,
                        const ArtifactStore &store) {
  ArtifactRootReference root{implementationPlatformSchema.identity.str(),
                             implementationPlatformSchema.version,
                             reference.artifact};
  auto platform = importImplementationPlatform(root, store);
  if (!platform)
    return platform.takeError();
  const TechnologyCorner *corner =
      platform->platform().findTechnologyCorner(reference.entity);
  if (!corner)
    return invalid("technology corner reference is out of range");
  return *corner;
}

} // namespace loom::platform
