#include "Common/IndexWidth.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <system_error>

namespace loom {

namespace {

constexpr std::uint64_t kDefaultIndexWidth = 32;

// The configured index width, read and parsed exactly once. `text` is a
// positive decimal override as written, empty when none applies. `width` is
// its value, absent only when the decimal is too large for a host integer to
// hold: an override that no host integer can represent is still an override,
// so it reaches the checked resolver instead of narrowing into a legal width.
struct ConfiguredIndexWidth {
  std::string text;
  std::optional<std::uint64_t> width = kDefaultIndexWidth;
};

const ConfiguredIndexWidth &configuredIndexWidth() {
  static const ConfiguredIndexWidth parsed = []() -> ConfiguredIndexWidth {
    const char *env = std::getenv("LOOM_INDEX_WIDTH");
    if (!env)
      return {};
    llvm::StringRef text(env);
    // Only a nonempty decimal digit sequence is an override, so a sign,
    // surrounding space, or trailing text leaves the default in place.
    if (text.empty() || !llvm::all_of(text, llvm::isDigit))
      return {};
    std::uint64_t value = 0;
    if (text.getAsInteger(10, value))
      return {text.str(), std::nullopt};
    if (value == 0)
      return {};
    return {text.str(), value};
  }();
  return parsed;
}

} // namespace

unsigned getIndexWidth() {
  // The narrow projection of that one configured fact, for the port-width
  // consumers that carry an `unsigned`. An override this type cannot
  // represent is not applied here rather than truncated into a legal width;
  // `getIndexBitWidth` reports it instead.
  const ConfiguredIndexWidth &configured = configuredIndexWidth();
  if (configured.width &&
      *configured.width <= std::numeric_limits<unsigned>::max())
    return static_cast<unsigned>(*configured.width);
  return static_cast<unsigned>(kDefaultIndexWidth);
}

namespace {

// The closest enclosing explicit declaration of the `index` width. Nested
// scopes are not a policy here: the innermost declaration is the one the data
// layout resolves, so it is also the one validated below.
mlir::DataLayoutEntryInterface findIndexLayoutEntry(mlir::Operation *op) {
  for (mlir::Operation *scope = op; scope; scope = scope->getParentOp()) {
    mlir::DataLayoutSpecInterface spec;
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(scope))
      spec = module.getDataLayoutSpec();
    else if (auto layoutOp = mlir::dyn_cast<mlir::DataLayoutOpInterface>(scope))
      spec = layoutOp.getDataLayoutSpec();
    if (!spec)
      continue;
    mlir::DataLayoutEntryList entries = spec.getSpecForType<mlir::IndexType>();
    if (!entries.empty())
      return entries.front();
  }
  return {};
}

llvm::Error unrepresentableWidth(llvm::StringRef width) {
  return llvm::createStringError(std::errc::value_too_large,
                                 "index bit width %s has no fixed "
                                 "representation",
                                 width.str().c_str());
}

// The one admissibility rule every source of the width answers to.
// `IntegerType::kMaxWidth` bounds every fixed integer representation, so a
// wider value has none and must never reach the type builder.
llvm::Expected<unsigned> checkFixedIndexWidth(std::uint64_t width) {
  if (width == 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "index bit width must be nonzero");
  if (width > mlir::IntegerType::kMaxWidth)
    return unrepresentableWidth(std::to_string(width));
  return static_cast<unsigned>(width);
}

} // namespace

llvm::Expected<unsigned> getIndexBitWidth(mlir::Operation *op) {
  mlir::DataLayoutEntryInterface entry =
      op ? findIndexLayoutEntry(op) : mlir::DataLayoutEntryInterface{};
  if (!entry) {
    const ConfiguredIndexWidth &configured = configuredIndexWidth();
    if (!configured.width)
      return unrepresentableWidth(configured.text);
    return checkFixedIndexWidth(*configured.width);
  }

  // The declaration is checked in its own integer representation first. The
  // data-layout query below resolves an index through `IntegerType::get`,
  // whose unsigned width would otherwise silently drop the high bits of a
  // larger declaration, or build a type past the fixed integer limit.
  auto declared = llvm::dyn_cast<mlir::IntegerAttr>(entry.getValue());
  if (!declared)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "index data layout entry does not declare a fixed integer width");
  const llvm::APInt &width = declared.getValue();
  // The active-bit test proves the value is a host integer before it is read
  // as one.
  if (width.getActiveBits() > 64) {
    llvm::SmallString<48> text;
    width.toString(text, 10, /*Signed=*/false);
    return unrepresentableWidth(text);
  }
  if (llvm::Expected<unsigned> admitted =
          checkFixedIndexWidth(width.getZExtValue());
      !admitted)
    return admitted.takeError();

  // A valid declaration is resolved by normal data-layout semantics, which a
  // dialect may implement itself, so its answer meets the same rule.
  llvm::TypeSize resolved = mlir::DataLayout::closest(op).getTypeSizeInBits(
      mlir::IndexType::get(op->getContext()));
  if (resolved.isScalable())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "index bit width must be a fixed width");
  return checkFixedIndexWidth(resolved.getFixedValue());
}

} // namespace loom
