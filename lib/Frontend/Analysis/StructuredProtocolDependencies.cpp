#include "Frontend/Analysis/StructuredProtocolDependencies.h"

#include "Frontend/Analysis/MemoryProvenance.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace loom::frontend::analysis {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_protocol_dependency_invalid: " +
                                     message);
}

struct FormalMemoryAccess final {
  bool reads = false;
  bool writes = false;
};

struct ProtocolCallable final {
  StructuredEntityRef reference;
  mlir::LLVM::LLVMFuncOp function;
  std::vector<FormalMemoryAccess> formalAccesses;
};

struct ProtocolCall final {
  std::size_t callableOrdinal = 0;
  mlir::LLVM::CallOp call;
};

std::optional<std::uint64_t> constantUnsignedValue(mlir::Value value) {
  mlir::Attribute attribute;
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    attribute = constant.getValue();
  else if (auto constant = value.getDefiningOp<mlir::LLVM::ConstantOp>())
    attribute = constant.getValue();
  auto integer = llvm::dyn_cast_if_present<mlir::IntegerAttr>(attribute);
  if (!integer || integer.getValue().isNegative() ||
      integer.getValue().getActiveBits() > 64)
    return std::nullopt;
  return integer.getValue().getZExtValue();
}

std::optional<std::uint64_t> fixedTypeByteCount(mlir::Operation *scope,
                                                mlir::Type type) {
  llvm::TypeSize size = mlir::DataLayout::closest(scope).getTypeSize(type);
  if (size.isScalable() || size.getFixedValue() == 0)
    return std::nullopt;
  return size.getFixedValue();
}

std::optional<std::uint64_t> fixedShapedByteCount(mlir::Operation *scope,
                                                  mlir::ShapedType type) {
  if (!type.hasStaticShape())
    return std::nullopt;
  std::uint64_t elements = 1;
  for (std::int64_t extent : type.getShape()) {
    if (extent <= 0 || elements > std::numeric_limits<std::uint64_t>::max() /
                                      static_cast<std::uint64_t>(extent))
      return std::nullopt;
    elements *= static_cast<std::uint64_t>(extent);
  }
  std::optional<std::uint64_t> elementBytes =
      fixedTypeByteCount(scope, type.getElementType());
  if (!elementBytes ||
      elements > std::numeric_limits<std::uint64_t>::max() / *elementBytes)
    return std::nullopt;
  return elements * *elementBytes;
}

std::optional<std::uint64_t>
fixedMemoryObjectByteCount(mlir::Value root,
                           mlir::SymbolTableCollection &symbols) {
  root = projectMemoryDerivationRoot(root);
  if (auto allocation = root.getDefiningOp<mlir::LLVM::AllocaOp>()) {
    std::optional<std::uint64_t> elements =
        constantUnsignedValue(allocation.getArraySize());
    std::optional<std::uint64_t> elementBytes =
        fixedTypeByteCount(allocation.getOperation(), allocation.getElemType());
    if (!elements || *elements == 0 || !elementBytes ||
        *elements > std::numeric_limits<std::uint64_t>::max() / *elementBytes)
      return std::nullopt;
    return *elements * *elementBytes;
  }
  if (auto allocation = root.getDefiningOp<mlir::memref::AllocOp>())
    return fixedShapedByteCount(allocation.getOperation(),
                                allocation.getType());
  if (auto allocation = root.getDefiningOp<mlir::memref::AllocaOp>())
    return fixedShapedByteCount(allocation.getOperation(),
                                allocation.getType());
  if (auto address = root.getDefiningOp<mlir::LLVM::AddressOfOp>()) {
    auto global = symbols.lookupNearestSymbolFrom<mlir::LLVM::GlobalOp>(
        address, address.getGlobalNameAttr());
    if (global)
      return fixedTypeByteCount(global.getOperation(), global.getGlobalType());
  }
  if (auto address = root.getDefiningOp<mlir::memref::GetGlobalOp>()) {
    auto global = symbols.lookupNearestSymbolFrom<mlir::memref::GlobalOp>(
        address, address.getNameAttr());
    if (global)
      return fixedShapedByteCount(global.getOperation(), global.getType());
  }
  return std::nullopt;
}

std::optional<unsigned> formalOrdinal(mlir::Value value,
                                      mlir::LLVM::LLVMFuncOp function) {
  value = projectMemoryDerivationRoot(value);
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!argument || function.isExternal() || function.getBody().empty() ||
      argument.getOwner() != &function.getBody().front() ||
      argument.getArgNumber() >= function.getFunctionType().getParams().size())
    return std::nullopt;
  return argument.getArgNumber();
}

std::vector<FormalMemoryAccess>
deriveFormalAccesses(mlir::LLVM::LLVMFuncOp function) {
  std::vector<FormalMemoryAccess> accesses(
      function.getFunctionType().getParams().size());
  function.walk([&](mlir::Operation *operation) {
    auto interface = llvm::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
    if (!interface)
      return;
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> effects;
    interface.getEffects(effects);
    for (const auto &effect : effects) {
      mlir::Value value = effect.getValue();
      if (!value)
        continue;
      std::optional<unsigned> ordinal = formalOrdinal(value, function);
      if (!ordinal)
        continue;
      FormalMemoryAccess &access = accesses[*ordinal];
      access.reads |= llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect());
      access.writes |=
          llvm::isa<mlir::MemoryEffects::Write>(effect.getEffect());
    }
  });
  return accesses;
}

bool precedes(mlir::Operation *producer, mlir::Operation *consumer) {
  return producer && consumer && producer->getBlock() == consumer->getBlock() &&
         producer->isBeforeInBlock(consumer);
}

} // namespace

llvm::Expected<std::vector<StructuredProtocolDependency>>
projectStructuredProtocolDependencies(
    const StructuredProgramCandidate &program,
    llvm::ArrayRef<StructuredEntityRef> protocolRoots) {
  auto view = program.view();
  if (!view)
    return view.takeError();

  std::vector<ProtocolCallable> callables;
  callables.reserve(protocolRoots.size());
  llvm::DenseSet<mlir::Operation *> uniqueFunctions;
  for (const StructuredEntityRef &reference : protocolRoots) {
    auto entity = view->resolve(reference);
    if (!entity)
      return entity.takeError();
    auto function =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity->operation);
    if (!function || function.isExternal())
      return invalid("protocol root is not a defined LLVM callable");
    if (!uniqueFunctions.insert(function.getOperation()).second)
      return invalid("protocol roots contain a duplicate callable");
    callables.push_back({reference, function, deriveFormalAccesses(function)});
  }

  mlir::SymbolTableCollection symbols;
  std::vector<ProtocolCall> calls;
  program.module().walk([&](mlir::LLVM::CallOp call) {
    if (!call.getCalleeAttr())
      return;
    auto callee = symbols.lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
        call, call.getCalleeAttr());
    if (!callee)
      return;
    for (std::size_t ordinal = 0; ordinal != callables.size(); ++ordinal)
      if (callables[ordinal].function == callee) {
        calls.push_back({ordinal, call});
        return;
      }
  });

  std::vector<StructuredProtocolDependency> dependencies;
  for (std::size_t producerOrdinal = 0; producerOrdinal != callables.size();
       ++producerOrdinal) {
    for (std::size_t consumerOrdinal = 0; consumerOrdinal != callables.size();
         ++consumerOrdinal) {
      if (producerOrdinal == consumerOrdinal)
        continue;
      llvm::DenseSet<mlir::Value> sharedRoots;
      for (ProtocolCall &producerCall : calls) {
        if (producerCall.callableOrdinal != producerOrdinal)
          continue;
        for (ProtocolCall &consumerCall : calls) {
          if (consumerCall.callableOrdinal != consumerOrdinal ||
              !precedes(producerCall.call, consumerCall.call))
            continue;
          auto producerArguments = producerCall.call.getArgOperands();
          auto consumerArguments = consumerCall.call.getArgOperands();
          const ProtocolCallable &producer = callables[producerOrdinal];
          const ProtocolCallable &consumer = callables[consumerOrdinal];
          for (std::size_t producerArgument = 0;
               producerArgument != producer.formalAccesses.size() &&
               producerArgument != producerArguments.size();
               ++producerArgument) {
            if (!producer.formalAccesses[producerArgument].writes)
              continue;
            mlir::Value producerRoot = projectMemoryDerivationRoot(
                producerArguments[producerArgument]);
            for (std::size_t consumerArgument = 0;
                 consumerArgument != consumer.formalAccesses.size() &&
                 consumerArgument != consumerArguments.size();
                 ++consumerArgument) {
              if (!consumer.formalAccesses[consumerArgument].reads)
                continue;
              mlir::Value consumerRoot = projectMemoryDerivationRoot(
                  consumerArguments[consumerArgument]);
              if (producerRoot == consumerRoot)
                sharedRoots.insert(producerRoot);
            }
          }
        }
      }
      if (!sharedRoots.empty()) {
        std::uint64_t knownBytes = 0;
        std::uint64_t unknownObjects = 0;
        for (mlir::Value root : sharedRoots) {
          std::optional<std::uint64_t> bytes =
              fixedMemoryObjectByteCount(root, symbols);
          if (!bytes) {
            ++unknownObjects;
            continue;
          }
          if (knownBytes > std::numeric_limits<std::uint64_t>::max() - *bytes)
            return invalid("protocol dependency byte count overflows");
          knownBytes += *bytes;
        }
        dependencies.push_back({callables[producerOrdinal].reference,
                                callables[consumerOrdinal].reference,
                                sharedRoots.size(), knownBytes,
                                unknownObjects});
      }
    }
  }
  return dependencies;
}

} // namespace loom::frontend::analysis
