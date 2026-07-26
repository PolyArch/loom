#ifndef LOOM_ADG_BUILDER_H
#define LOOM_ADG_BUILDER_H

#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricEnums.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "mlir/IR/Value.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace loom::adg {
namespace detail {
class DesignState;
}

/// A typed authoring description of one legal fabric.module port type.
class PortType final {
public:
  enum class Kind : std::uint8_t { Bits, TaggedBits, Memory };

  static llvm::Expected<PortType> bits(std::uint32_t width);
  static llvm::Expected<PortType> taggedBits(std::uint32_t width,
                                             std::uint32_t tagWidth);
  static llvm::Expected<PortType> memory(llvm::ArrayRef<std::int64_t> shape,
                                         const PortType &elementType);

  Kind kind() const { return kind_; }
  std::uint32_t width() const { return width_; }
  std::uint32_t tagWidth() const { return tagWidth_; }
  llvm::ArrayRef<std::int64_t> shape() const { return shape_; }

  friend bool operator==(const PortType &left, const PortType &right) {
    return left.kind_ == right.kind_ && left.width_ == right.width_ &&
           left.tagWidth_ == right.tagWidth_ && left.shape_ == right.shape_;
  }

private:
  PortType(Kind kind, std::uint32_t width, std::uint32_t tagWidth,
           std::vector<std::int64_t> shape)
      : kind_(kind), width_(width), tagWidth_(tagWidth),
        shape_(std::move(shape)) {}

  Kind kind_;
  std::uint32_t width_;
  std::uint32_t tagWidth_;
  std::vector<std::int64_t> shape_;
};

/// An owner-checked SSA transport or memory capability in one SpatialCore.
class SpatialValue final {
public:
  SpatialValue() = default;

private:
  SpatialValue(const std::shared_ptr<detail::DesignState> &state,
               std::size_t rootOrdinal, mlir::Value value)
      : state_(state), rootOrdinal_(rootOrdinal), value_(value) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  mlir::Value value_;

  friend class SpatialCoreBuilder;
};

struct FifoSpec final {
  PortType outputType;
  std::uint32_t maxDepth;
  bool bypassable;
};

struct BoundarySpec final {
  ::fabric::BoundaryDirection direction;
  std::vector<PortType> inputTypes;
  std::vector<PortType> outputTypes;

  static BoundarySpec s2t(const PortType &dataInput, const PortType &tagInput,
                          const PortType &taggedOutput);
  static BoundarySpec t2s(const PortType &taggedInput,
                          llvm::ArrayRef<PortType> outputs);
};

class SpatialCoreBuilder final {
public:
  llvm::Expected<SpatialValue> input(std::size_t ordinal) const;

  llvm::Expected<SpatialValue> addFifo(SpatialValue input,
                                       const FifoSpec &spec);

  llvm::Expected<std::vector<SpatialValue>>
  addBoundary(llvm::ArrayRef<SpatialValue> inputs, const BoundarySpec &spec);

  /// Closes this root with the exact declared result sequence.
  llvm::Error close(llvm::ArrayRef<SpatialValue> outputs);

private:
  llvm::Expected<mlir::Value>
  resolveValue(const std::shared_ptr<detail::DesignState> &state,
               const SpatialValue &value) const;

  SpatialCoreBuilder(const std::shared_ptr<detail::DesignState> &state,
                     std::size_t rootOrdinal)
      : state_(state), rootOrdinal_(rootOrdinal) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_;

  friend class DesignBuilder;
};

/// Immutable transient closure over finalized Fabric roots.
class FinalizedFabricDesign final {
public:
  llvm::ArrayRef<loom::fabric::FinalizedFabricRoot> roots() const {
    return roots_;
  }

private:
  explicit FinalizedFabricDesign(
      std::vector<loom::fabric::FinalizedFabricRoot> roots)
      : roots_(std::move(roots)) {}

  std::vector<loom::fabric::FinalizedFabricRoot> roots_;

  friend class DesignBuilder;
};

/// The sole owner of one typed Fabric authoring draft.
class DesignBuilder final {
public:
  explicit DesignBuilder(const loom::ArtifactStore &store);
  ~DesignBuilder();

  DesignBuilder(const DesignBuilder &) = delete;
  DesignBuilder &operator=(const DesignBuilder &) = delete;
  DesignBuilder(DesignBuilder &&) noexcept;
  DesignBuilder &operator=(DesignBuilder &&) noexcept;

  llvm::Expected<SpatialCoreBuilder>
  createSpatialCore(llvm::StringRef label, llvm::ArrayRef<PortType> inputs,
                    llvm::ArrayRef<PortType> outputs);

  llvm::Expected<FinalizedFabricDesign> finalize() &&;

private:
  std::shared_ptr<detail::DesignState> state_;
};

} // namespace loom::adg

#endif // LOOM_ADG_BUILDER_H
