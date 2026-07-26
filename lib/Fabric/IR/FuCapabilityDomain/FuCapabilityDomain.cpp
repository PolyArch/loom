#include "Fabric/IR/FuCapabilityDomain.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <limits>
#include <system_error>

namespace fabric {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(std::errc::invalid_argument,
                                 "invalid FuCapabilityDomainRecord: %s",
                                 message.str().c_str());
}

class Writer final {
public:
  void u64(std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
  }

  llvm::Error count(std::size_t value, llvm::StringRef field) {
    if (value > std::numeric_limits<std::uint64_t>::max())
      return invalid(field + " count exceeds u64");
    u64(static_cast<std::uint64_t>(value));
    return llvm::Error::success();
  }

  void bytes(llvm::ArrayRef<std::uint8_t> values) {
    bytes_.insert(bytes_.end(), values.begin(), values.end());
  }

  std::vector<std::uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<std::uint8_t> bytes_;
};

class Reader final {
public:
  explicit Reader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint64_t> u64(llvm::StringRef field) {
    if (remaining() < 8)
      return invalid("truncated " + field);
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  std::size_t remaining() const { return bytes_.size() - offset_; }

  llvm::Error finish() const {
    return remaining() == 0 ? llvm::Error::success()
                            : invalid("record has trailing bytes");
  }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

llvm::Expected<std::vector<std::uint8_t>>
encodeTemplate(const FuCapabilityTemplateSelection &selection) {
  Writer writer;
  if (llvm::Error error = writer.count(
          selection.activeOperationNodeOrdinals.size(), "active operations"))
    return std::move(error);
  for (std::uint64_t operation : selection.activeOperationNodeOrdinals)
    writer.u64(operation);
  if (llvm::Error error = writer.count(selection.routes.size(), "routes"))
    return std::move(error);
  for (const FuCapabilityRouteSelection &route : selection.routes) {
    writer.u64(route.selectorNodeOrdinal);
    writer.u64(route.selectedPort);
  }
  return writer.take();
}

llvm::Expected<std::vector<FuCapabilityTemplateSelection>>
normalize(std::vector<FuCapabilityTemplateSelection> templates) {
  if (templates.empty())
    return invalid("template domain is empty");

  using KeyedTemplate =
      std::pair<std::vector<std::uint8_t>, FuCapabilityTemplateSelection>;
  std::vector<KeyedTemplate> keyed;
  keyed.reserve(templates.size());
  for (FuCapabilityTemplateSelection &selection : templates) {
    if (selection.activeOperationNodeOrdinals.empty())
      return invalid("template has no active fabric.op");
    llvm::sort(selection.activeOperationNodeOrdinals);
    if (std::adjacent_find(selection.activeOperationNodeOrdinals.begin(),
                           selection.activeOperationNodeOrdinals.end()) !=
        selection.activeOperationNodeOrdinals.end())
      return invalid("template repeats an active operation");

    llvm::sort(selection.routes, [](const FuCapabilityRouteSelection &left,
                                    const FuCapabilityRouteSelection &right) {
      if (left.selectorNodeOrdinal != right.selectorNodeOrdinal)
        return left.selectorNodeOrdinal < right.selectorNodeOrdinal;
      return left.selectedPort < right.selectedPort;
    });
    for (std::size_t index = 1; index < selection.routes.size(); ++index)
      if (selection.routes[index - 1].selectorNodeOrdinal ==
          selection.routes[index].selectorNodeOrdinal)
        return invalid("template selects one routing node more than once");

    auto bytes = encodeTemplate(selection);
    if (!bytes)
      return bytes.takeError();
    keyed.emplace_back(std::move(*bytes), std::move(selection));
  }

  llvm::sort(keyed, [](const KeyedTemplate &left, const KeyedTemplate &right) {
    return left.first < right.first;
  });
  for (std::size_t index = 1; index < keyed.size(); ++index)
    if (keyed[index - 1].first == keyed[index].first)
      return invalid("template domain contains a duplicate row");

  templates.clear();
  templates.reserve(keyed.size());
  for (KeyedTemplate &entry : keyed)
    templates.push_back(std::move(entry.second));
  return templates;
}

} // namespace

llvm::Expected<FuCapabilityDomainRecord> FuCapabilityDomainRecord::create(
    std::vector<FuCapabilityTemplateSelection> templates) {
  auto normalized = normalize(std::move(templates));
  if (!normalized)
    return normalized.takeError();
  return FuCapabilityDomainRecord(std::move(*normalized));
}

llvm::Expected<FuCapabilityDomainRecord>
FuCapabilityDomainRecord::fromCanonical(
    std::vector<FuCapabilityTemplateSelection> templates) {
  const std::vector<FuCapabilityTemplateSelection> original = templates;
  auto normalized = normalize(std::move(templates));
  if (!normalized)
    return normalized.takeError();
  if (original != *normalized)
    return invalid("fields are not in canonical order");
  return FuCapabilityDomainRecord(std::move(*normalized));
}

llvm::Expected<std::vector<std::uint8_t>>
encodeFuCapabilityDomainRecord(const FuCapabilityDomainRecord &record) {
  Writer writer;
  if (llvm::Error error = writer.count(record.templates().size(), "templates"))
    return std::move(error);
  for (const FuCapabilityTemplateSelection &selection : record.templates()) {
    auto bytes = encodeTemplate(selection);
    if (!bytes)
      return bytes.takeError();
    writer.bytes(*bytes);
  }
  return writer.take();
}

llvm::Expected<FuCapabilityDomainRecord>
decodeFuCapabilityDomainRecord(llvm::ArrayRef<std::uint8_t> bytes) {
  Reader reader(bytes);
  auto templateCount = reader.u64("template count");
  if (!templateCount)
    return templateCount.takeError();
  if (*templateCount == 0 || *templateCount > reader.remaining() / 16)
    return invalid("template count exceeds its framing");

  std::vector<FuCapabilityTemplateSelection> templates;
  templates.reserve(*templateCount);
  for (std::uint64_t templateOrdinal = 0; templateOrdinal < *templateCount;
       ++templateOrdinal) {
    FuCapabilityTemplateSelection selection;
    auto operationCount = reader.u64("active operation count");
    if (!operationCount)
      return operationCount.takeError();
    if (*operationCount == 0 || *operationCount > reader.remaining() / 8)
      return invalid("active operation count exceeds its framing");
    selection.activeOperationNodeOrdinals.reserve(*operationCount);
    for (std::uint64_t index = 0; index < *operationCount; ++index) {
      auto operation = reader.u64("active operation ordinal");
      if (!operation)
        return operation.takeError();
      selection.activeOperationNodeOrdinals.push_back(*operation);
    }

    auto routeCount = reader.u64("route count");
    if (!routeCount)
      return routeCount.takeError();
    if (*routeCount > reader.remaining() / 16)
      return invalid("route count exceeds its framing");
    selection.routes.reserve(*routeCount);
    for (std::uint64_t index = 0; index < *routeCount; ++index) {
      auto selector = reader.u64("selector node ordinal");
      auto port = reader.u64("selected port ordinal");
      if (!selector)
        return selector.takeError();
      if (!port)
        return port.takeError();
      selection.routes.push_back({*selector, *port});
    }
    templates.push_back(std::move(selection));
  }
  if (llvm::Error error = reader.finish())
    return std::move(error);

  auto record = FuCapabilityDomainRecord::fromCanonical(std::move(templates));
  if (!record)
    return record.takeError();
  auto canonical = encodeFuCapabilityDomainRecord(*record);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return invalid("record is not canonical");
  return record;
}

} // namespace fabric
