#include "Fabric/IR/StreamConfiguration.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

namespace fabric {
namespace {

using mlir::ArrayAttr;
using mlir::Attribute;
using mlir::DictionaryAttr;
using mlir::FlatSymbolRefAttr;
using mlir::IntegerAttr;
using mlir::IntegerType;

bool isSingletonStream(OpOp op) {
  auto operations = op.getOpList();
  if (operations.size() != 1)
    return false;
  auto symbol = mlir::dyn_cast<FlatSymbolRefAttr>(operations[0]);
  return symbol && symbol.getValue() == "dataflow.stream";
}

std::optional<mlir::arith::CmpIPredicate> predicateFromAttr(Attribute attr) {
  if (auto predicate =
          mlir::dyn_cast_or_null<mlir::arith::CmpIPredicateAttr>(attr))
    return predicate.getValue();
  auto integer = mlir::dyn_cast_or_null<IntegerAttr>(attr);
  if (!integer)
    return std::nullopt;
  auto type = mlir::dyn_cast<IntegerType>(integer.getType());
  if (!type || !type.isSignless() || type.getWidth() != 64)
    return std::nullopt;
  return mlir::arith::symbolizeCmpIPredicate(
      static_cast<std::uint64_t>(integer.getValue().getZExtValue()));
}

std::string printAttribute(Attribute attr) {
  std::string text;
  llvm::raw_string_ostream os(text);
  attr.print(os);
  return text;
}

void appendPredicate(StreamConfiguration &config,
                     mlir::arith::CmpIPredicate predicate) {
  if (!config.supports(predicate))
    config.predicates.push_back(predicate);
}

mlir::FailureOr<StreamConfiguration>
parseLegacyConfiguration(OpOp op, std::string &error) {
  ArrayAttr hwParams = op.getHwParamsAttr();
  if (!hwParams || hwParams.size() != 1) {
    error =
        "@dataflow.stream requires exactly one fixed 'step_kind' capability";
    return mlir::failure();
  }
  auto hardware = mlir::dyn_cast<DictionaryAttr>(hwParams[0]);
  if (!hardware) {
    error = "@dataflow.stream hw_params must be a dictionary";
    return mlir::failure();
  }
  for (mlir::NamedAttribute field : hardware) {
    llvm::StringRef name = field.getName().getValue();
    if (name != "step_kind" && name != "predicate") {
      error = "@dataflow.stream legacy hw_params only supports 'step_kind' "
              "and 'predicate'";
      return mlir::failure();
    }
  }

  auto stepKind =
      dataflow::getStreamStepKindFromAttr(hardware.get("step_kind"));
  if (!stepKind) {
    error =
        "@dataflow.stream requires exactly one fixed 'step_kind' capability";
    return mlir::failure();
  }

  auto predicates = hardware.getAs<ArrayAttr>("predicate");
  if (!predicates || predicates.empty()) {
    error = "@dataflow.stream requires a non-empty 'predicate' capability "
            "set";
    return mlir::failure();
  }

  StreamConfiguration config{*stepKind, {}, std::nullopt};
  llvm::SmallSet<mlir::arith::CmpIPredicate, 16> seen;
  for (Attribute attr : predicates) {
    auto predicate = predicateFromAttr(attr);
    if (!predicate) {
      error = "@dataflow.stream 'predicate' entries must be arith.cmpi "
              "predicate enum attrs";
      return mlir::failure();
    }
    if (!seen.insert(*predicate).second) {
      error = "@dataflow.stream 'predicate' capability set contains a "
              "duplicate";
      return mlir::failure();
    }
    config.predicates.push_back(*predicate);
  }

  DictionaryAttr software = op.getSwConfigsAttr();
  if (!software)
    return config;
  if (software.get("step_kind")) {
    error = "@dataflow.stream must not select 'step_kind' through sw_configs";
    return mlir::failure();
  }
  for (mlir::NamedAttribute field : software) {
    if (field.getName().getValue() != "predicate") {
      error = "@dataflow.stream sw_configs only supports 'predicate'";
      return mlir::failure();
    }
  }
  if (Attribute selected = software.get("predicate")) {
    auto predicate = predicateFromAttr(selected);
    if (!predicate) {
      error = "@dataflow.stream sw_configs.predicate must be an arith.cmpi "
              "predicate enum attr";
      return mlir::failure();
    }
    if (!config.supports(*predicate)) {
      error = "'sw_configs[\"predicate\"]' value " + printAttribute(selected) +
              " is not in the 'hw_params[\"predicate\"]' allowed set";
      return mlir::failure();
    }
    config.selectedPredicate = *predicate;
  }
  return config;
}

mlir::FailureOr<StreamConfiguration>
parseNormalizedConfiguration(OpOp op, std::string &error) {
  ArrayAttr modes = op.getHwParamsAttr();
  std::optional<dataflow::StreamStepKind> fixedStep;
  llvm::SmallVector<mlir::arith::CmpIPredicate, 4> modePredicates;
  StreamConfiguration config{dataflow::StreamStepKind::Add, {}, std::nullopt};
  for (auto [modeIndex, attr] : llvm::enumerate(modes)) {
    auto mode = mlir::dyn_cast<DictionaryAttr>(attr);
    auto operation = mode ? mode.getAs<FlatSymbolRefAttr>("op") : nullptr;
    auto attributes = mode ? mode.getAs<DictionaryAttr>("attributes") : nullptr;
    if (!operation || operation.getValue() != "dataflow.stream" ||
        !attributes) {
      error = "normalized @dataflow.stream mode must select "
              "@dataflow.stream and provide attributes";
      return mlir::failure();
    }
    auto stepKind =
        dataflow::getStreamStepKindFromAttr(attributes.get("step_kind"));
    if (!stepKind) {
      error = "hw_params mode #" + std::to_string(modeIndex) +
              " has invalid dataflow.stream step_kind";
      return mlir::failure();
    }
    auto predicate = predicateFromAttr(attributes.get("predicate"));
    if (!predicate) {
      error = "hw_params mode #" + std::to_string(modeIndex) +
              " has invalid dataflow.stream predicate";
      return mlir::failure();
    }
    if (fixedStep && *fixedStep != *stepKind) {
      error = "@dataflow.stream hw_params modes must share one fixed "
              "'step_kind'";
      return mlir::failure();
    }
    fixedStep = *stepKind;
    config.stepKind = *stepKind;
    appendPredicate(config, *predicate);
    modePredicates.push_back(*predicate);
  }

  DictionaryAttr software = op.getSwConfigsAttr();
  if (!software)
    return config;
  if (software.size() != 1 || !software.get("mode")) {
    error = "normalized hw_params requires sw_configs = {mode = N}";
    return mlir::failure();
  }
  auto selected = mlir::dyn_cast<IntegerAttr>(software.get("mode"));
  if (!selected || selected.getValue().isNegative() ||
      selected.getValue().getActiveBits() > 32) {
    error = "'sw_configs.mode' must be a non-negative i32";
    return mlir::failure();
  }
  std::uint64_t modeIndex = selected.getValue().getZExtValue();
  if (modeIndex >= modePredicates.size()) {
    error = "'sw_configs.mode' is out of range for hw_params";
    return mlir::failure();
  }
  config.selectedPredicate = modePredicates[modeIndex];
  return config;
}

} // namespace

bool StreamConfiguration::supports(mlir::arith::CmpIPredicate predicate) const {
  return llvm::is_contained(predicates, predicate);
}

mlir::FailureOr<StreamConfiguration>
parseStreamConfiguration(OpOp op, std::string &error) {
  if (!isSingletonStream(op)) {
    error = "stream configuration requires op_list [@dataflow.stream]";
    return mlir::failure();
  }
  FabricOpModeClassification classification = classifyFabricOpModes(op);
  if (classification.kind == FabricOpModeKind::Malformed) {
    error = std::move(classification.diagnostic);
    return mlir::failure();
  }
  if (classification.kind == FabricOpModeKind::Normalized)
    return parseNormalizedConfiguration(op, error);
  return parseLegacyConfiguration(op, error);
}

} // namespace fabric
