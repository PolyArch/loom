#ifndef FABRIC_IR_CONFIGUREDFUNCTION_H
#define FABRIC_IR_CONFIGUREDFUNCTION_H

#include "Fabric/IR/FabricOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>
#include <utility>

namespace fabric {

struct ConfiguredValue {
  enum class Kind : unsigned char { InputPort, NodeResult };

  Kind kind = Kind::InputPort;
  unsigned index = 0;
  unsigned result = 0;

  static ConfiguredValue input(unsigned port) {
    return {Kind::InputPort, port, 0};
  }
  static ConfiguredValue nodeResult(unsigned node, unsigned result) {
    return {Kind::NodeResult, node, result};
  }

  bool operator==(const ConfiguredValue &other) const {
    return kind == other.kind && index == other.index &&
           (kind == Kind::InputPort || result == other.result);
  }
};

struct ConfiguredBoundaryInput {
  unsigned fuPort = 0;
  ::mlir::Type type;
};

struct ConfiguredFunctionNode {
  unsigned fabricResource = 0;
  std::string operationName;
  ::mlir::FunctionType functionType;
  ::mlir::DictionaryAttr attributes;
  ::llvm::SmallVector<ConfiguredValue, 4> operands;
};

struct ConfiguredBoundaryOutput {
  unsigned fuPort = 0;
  ::mlir::Type type;
  ConfiguredValue value;
};

struct ConfiguredFunction {
  ::llvm::SmallVector<ConfiguredBoundaryInput, 4> inputs;
  ::llvm::SmallVector<ConfiguredFunctionNode, 8> nodes;
  ::llvm::SmallVector<ConfiguredBoundaryOutput, 4> outputs;
};

struct ConfiguredFunctionMatch {
  ::llvm::SmallVector<unsigned, 8> nodeMap;
  ::llvm::SmallVector<std::pair<unsigned, unsigned>, 4> inputPorts;
  ::llvm::SmallVector<std::pair<unsigned, unsigned>, 4> outputPorts;
};

struct ConfiguredFunctionKey {
  std::uint64_t hash = 0;
  std::string canonical;
};

::mlir::LogicalResult projectConfiguredFunction(FuOp fu,
                                                ::mlir::DictionaryAttr encoding,
                                                ConfiguredFunction &function,
                                                std::string &error);

::mlir::LogicalResult projectConfiguredFunctions(
    FuOp fu, ::llvm::SmallVectorImpl<ConfiguredFunction> &functions,
    std::string &error);

bool matchConfiguredFunctions(const ConfiguredFunction &pattern,
                              const ConfiguredFunction &candidate,
                              bool preserveFuBoundaryIdentity,
                              ConfiguredFunctionMatch *witness = nullptr);

ConfiguredFunctionKey
getConfiguredFunctionKey(const ConfiguredFunction &function,
                         bool preserveFuBoundaryIdentity);

::mlir::LogicalResult verifyValidSemanticEncodings(FuOp fu);

::mlir::LogicalResult verifyNormalizedHardwareModes(OpOp op);

unsigned getValidSemanticEncodingCount(FuOp fu);

} // namespace fabric

#endif // FABRIC_IR_CONFIGUREDFUNCTION_H
