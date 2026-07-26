#ifndef LOOM_ADG_BUILDER_H
#define LOOM_ADG_BUILDER_H

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricEnums.h"
#include "Fabric/IR/ImplementationFamily.h"
#include "Fabric/IR/MemoryOperationPort.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "mlir/IR/Value.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace loom::adg {
namespace detail {
class DesignState;
}

class FuBuilder;
class PeBuilder;

/// A typed authoring description of one legal fabric.module port type.
class PortType final {
public:
  enum class Kind : std::uint8_t { Bits, TaggedBits, Memory };

  static constexpr std::int64_t kDynamicExtent =
      std::numeric_limits<std::int64_t>::min();

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
  friend class PeBuilder;
};

/// An owner-checked value on one PE's internal untagged boundary.
class PeValue final {
public:
  PeValue() = default;

private:
  PeValue(const std::shared_ptr<detail::DesignState> &state,
          std::size_t rootOrdinal, std::size_t peOrdinal, mlir::Value value)
      : state_(state), rootOrdinal_(rootOrdinal), peOrdinal_(peOrdinal),
        value_(value) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  std::size_t peOrdinal_ = 0;
  mlir::Value value_;

  friend class PeBuilder;
};

/// An owner-checked value in one FU graph region.
class FuValue final {
public:
  FuValue() = default;

private:
  FuValue(const std::shared_ptr<detail::DesignState> &state,
          std::size_t rootOrdinal, std::size_t peOrdinal, std::size_t fuOrdinal,
          mlir::Value value)
      : state_(state), rootOrdinal_(rootOrdinal), peOrdinal_(peOrdinal),
        fuOrdinal_(fuOrdinal), value_(value) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_ = 0;
  std::size_t peOrdinal_ = 0;
  std::size_t fuOrdinal_ = 0;
  mlir::Value value_;

  friend class FuBuilder;
};

enum class FuConfigurationMode : std::uint8_t { PerInstruction, PerFu };

struct TemporalRegisterFifoParameters final {
  std::uint32_t count;
  std::uint32_t depth;
  std::uint32_t ports;
};

struct TemporalPeParameters final {
  std::uint32_t instructionCapacity;
  FuConfigurationMode fuConfigurationMode;
  ::fabric::OperandBufferMode operandBufferMode;
  std::uint32_t operandBufferSize;
  std::optional<TemporalRegisterFifoParameters> registerFifos;
};

/// A closed typed description of one anonymous PE boundary and schedule.
class PeSpec final {
public:
  static PeSpec spatial(std::vector<PortType> inputTypes,
                        std::vector<PortType> outputTypes);
  static PeSpec temporal(std::vector<PortType> inputTypes,
                         std::vector<PortType> outputTypes,
                         TemporalPeParameters parameters);

private:
  PeSpec(::fabric::Schedule schedule, std::vector<PortType> inputTypes,
         std::vector<PortType> outputTypes,
         std::optional<TemporalPeParameters> temporal)
      : schedule_(schedule), inputTypes_(std::move(inputTypes)),
        outputTypes_(std::move(outputTypes)), temporal_(std::move(temporal)) {}

  ::fabric::Schedule schedule_;
  std::vector<PortType> inputTypes_;
  std::vector<PortType> outputTypes_;
  std::optional<TemporalPeParameters> temporal_;

  friend class SpatialCoreBuilder;
};

struct FuSpec final {
  std::vector<PortType> inputTypes;
  std::vector<PortType> outputTypes;
};

/// One concrete fabric.op capability using the generated semantic owners.
struct OperationCapabilitySpec final {
  ::fabric::ImplementationFamilyId implementationFamily;
  ::fabric::FamilyCapabilityParams hardwareParameters;
  std::vector<::dataflow::OperationSchemaId> enabledOperations;
  std::vector<PortType> outputTypes;
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

struct SwitchSpec final {
  ::fabric::Schedule schedule;
  std::vector<PortType> inputTypes;
  std::vector<PortType> outputTypes;
  std::vector<std::vector<std::uint32_t>> sourcesByOutput;
  std::optional<std::uint32_t> routeTableSize;

  static SwitchSpec
  spatial(std::vector<PortType> inputTypes, std::vector<PortType> outputTypes,
          std::vector<std::vector<std::uint32_t>> sourcesByOutput);

  static SwitchSpec
  temporal(std::vector<PortType> inputTypes, std::vector<PortType> outputTypes,
           std::vector<std::vector<std::uint32_t>> sourcesByOutput,
           std::uint32_t routeTableSize);
};

/// One exact spatial fabric.mem Operation Engine declaration.
class MemorySpec final {
public:
  static MemorySpec
  spatial(std::vector<PortType> inputTypes, std::vector<PortType> outputTypes,
          std::vector<std::uint32_t> managerInputOrdinals,
          std::vector<std::uint32_t> subordinateOutputOrdinals,
          std::vector<::fabric::MemoryOperationPortDeclaration> operationPorts);

private:
  MemorySpec(
      std::vector<PortType> inputTypes, std::vector<PortType> outputTypes,
      std::vector<std::uint32_t> managerInputOrdinals,
      std::vector<std::uint32_t> subordinateOutputOrdinals,
      std::vector<::fabric::MemoryOperationPortDeclaration> operationPorts)
      : inputTypes_(std::move(inputTypes)),
        outputTypes_(std::move(outputTypes)),
        managerInputOrdinals_(std::move(managerInputOrdinals)),
        subordinateOutputOrdinals_(std::move(subordinateOutputOrdinals)),
        operationPorts_(std::move(operationPorts)) {}

  std::vector<PortType> inputTypes_;
  std::vector<PortType> outputTypes_;
  std::vector<std::uint32_t> managerInputOrdinals_;
  std::vector<std::uint32_t> subordinateOutputOrdinals_;
  std::vector<::fabric::MemoryOperationPortDeclaration> operationPorts_;

  friend class SpatialCoreBuilder;
};

class FuBuilder final {
public:
  llvm::Expected<FuValue> input(std::size_t ordinal) const;

  llvm::Expected<std::vector<FuValue>>
  addOperation(llvm::ArrayRef<FuValue> inputs,
               const OperationCapabilitySpec &spec);

  llvm::Expected<FuValue> addMux(llvm::ArrayRef<FuValue> inputs);

  llvm::Expected<std::vector<FuValue>> addDemux(FuValue input,
                                                std::uint32_t outputCount);

  llvm::Error close(llvm::ArrayRef<FuValue> outputs);

private:
  llvm::Expected<mlir::Value>
  resolveValue(const std::shared_ptr<detail::DesignState> &state,
               const FuValue &value) const;

  FuBuilder(const std::shared_ptr<detail::DesignState> &state,
            std::size_t rootOrdinal, std::size_t peOrdinal,
            std::size_t fuOrdinal)
      : state_(state), rootOrdinal_(rootOrdinal), peOrdinal_(peOrdinal),
        fuOrdinal_(fuOrdinal) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_;
  std::size_t peOrdinal_;
  std::size_t fuOrdinal_;

  friend class PeBuilder;
};

class PeBuilder final {
public:
  llvm::Expected<PeValue> input(std::size_t ordinal) const;
  llvm::Expected<SpatialValue> output(std::size_t ordinal) const;

  llvm::Expected<FuBuilder> addFu(llvm::ArrayRef<PeValue> inputs,
                                  const FuSpec &spec);

  llvm::Error close();

private:
  llvm::Expected<mlir::Value>
  resolveValue(const std::shared_ptr<detail::DesignState> &state,
               const PeValue &value) const;

  PeBuilder(const std::shared_ptr<detail::DesignState> &state,
            std::size_t rootOrdinal, std::size_t peOrdinal)
      : state_(state), rootOrdinal_(rootOrdinal), peOrdinal_(peOrdinal) {}

  std::weak_ptr<detail::DesignState> state_;
  std::size_t rootOrdinal_;
  std::size_t peOrdinal_;

  friend class SpatialCoreBuilder;
};

class SpatialCoreBuilder final {
public:
  llvm::Expected<SpatialValue> input(std::size_t ordinal) const;

  llvm::Expected<SpatialValue> addFifo(SpatialValue input,
                                       const FifoSpec &spec);

  llvm::Expected<std::vector<SpatialValue>>
  addBoundary(llvm::ArrayRef<SpatialValue> inputs, const BoundarySpec &spec);

  llvm::Expected<std::vector<SpatialValue>>
  addSwitch(llvm::ArrayRef<SpatialValue> inputs, const SwitchSpec &spec);

  llvm::Expected<std::vector<SpatialValue>>
  addMemory(llvm::ArrayRef<SpatialValue> inputs, const MemorySpec &spec);

  llvm::Expected<PeBuilder> addPe(llvm::ArrayRef<SpatialValue> inputs,
                                  const PeSpec &spec);

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
