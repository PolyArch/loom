//===- FabricMemOp.cpp - Parser/printer/verifier for fabric.mem -----------===//
//
// Implements the typed occurrence contract and the existing operation-engine
// port ABI for fabric.mem. Endpoint roles reference signature positions;
// function_type remains the sole owner of their MLIR kinds and widths.
//
//===----------------------------------------------------------------------===//

#include "Common/IndexWidth.h"
#include "Common/LoomConstants.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <limits>

using namespace mlir;
using namespace fabric;

namespace mlir {

template <>
LogicalResult
RegisteredOperationName::Model<::fabric::MemOp>::setPropertiesFromAttr(
    OperationName, PropertyRef properties, Attribute attr,
    function_ref<InFlightDiagnostic()> emitError) {
  auto *memProperties = properties.as<::fabric::MemOp::Properties *>();
  return ::fabric::MemOp::setPropertiesFromParsedAttr(*memProperties, attr,
                                                      emitError);
}

} // namespace mlir

namespace fabric {

unsigned resolveLoomAddrBits(Operation *op) {
  Operation *cur = op;
  while (cur) {
    if (auto module = dyn_cast<ModuleOp>(cur)) {
      if (auto attr = module.getLoomAddrBitsAttr())
        return static_cast<unsigned>(attr.getInt());
      break;
    }
    cur = cur->getParentOp();
  }
  return ::loom::getDefaultLoomAddrBits();
}

unsigned resolveLoomMemBusWidth(Operation *op) {
  Operation *cur = op;
  while (cur) {
    if (auto module = dyn_cast<ModuleOp>(cur)) {
      if (auto attr = module.getLoomMemBusWidthAttr())
        return static_cast<unsigned>(attr.getInt());
      break;
    }
    cur = cur->getParentOp();
  }
  return ::loom::getDefaultLoomMemBusWidth();
}

} // namespace fabric

namespace {

constexpr StringLiteral kLoadGroupSize = "load_group_size";
constexpr StringLiteral kStoreGroupSize = "store_group_size";
constexpr StringLiteral kDataWidth = "data_width";
constexpr StringLiteral kTagWidth = "tag_width";
constexpr StringLiteral kOperationTableSize = "operation_table_size";
constexpr StringLiteral kDispatchEligibility = "dispatch_eligibility";
constexpr StringLiteral kOperationPortRequests = "operation_port_requests";
constexpr StringLiteral kSubordinateRequests = "subordinate_requests";
constexpr StringLiteral kLocalMemoryService = "local_memory_service";

struct EngineInfo {
  unsigned loadCount = 0;
  unsigned storeCount = 0;
  unsigned dataWidth = 0;
  unsigned tagWidth = 0;
  unsigned operationTableSize = 0;
};

struct SignatureLayout {
  unsigned managerCount = 0;
  unsigned subordinateCount = 0;
  unsigned loadOperandBase = 0;
  unsigned storeOperandBase = 0;
  unsigned loadResultBase = 0;
  unsigned storeResultBase = 0;
};

static ParseResult parseFunctionType(OpAsmParser &parser,
                                     OperationState &result) {
  SmallVector<Type> inputs;
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseTypeList(inputs) || parser.parseRParen())
      return failure();
  }
  if (parser.parseArrow())
    return failure();

  SmallVector<Type> results;
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parser.parseOptionalRParen())) {
      if (parser.parseTypeList(results) || parser.parseRParen())
        return failure();
    }
  } else {
    Type type;
    if (parser.parseType(type))
      return failure();
    results.push_back(type);
  }

  result.addAttribute(
      "function_type",
      TypeAttr::get(FunctionType::get(parser.getContext(), inputs, results)));
  return success();
}

static ParseResult parseHardwareParameters(OpAsmParser &parser,
                                           OperationState &result) {
  if (failed(parser.parseOptionalLSquare()))
    return success();

  SmallVector<Attribute, 1> elements;
  auto parseElement = [&]() -> ParseResult {
    DictionaryAttr dictionary;
    if (parser.parseAttribute(dictionary))
      return failure();
    elements.push_back(dictionary);
    return success();
  };

  if (failed(parser.parseOptionalRSquare())) {
    if (parseElement())
      return failure();
    while (succeeded(parser.parseOptionalComma()))
      if (parseElement())
        return failure();
    if (parser.parseRSquare())
      return failure();
  }

  result.addAttribute("hw_params",
                      ArrayAttr::get(parser.getContext(), elements));
  return success();
}

static ParseResult parseDiscardableAttributes(OpAsmParser &parser,
                                              OperationState &result) {
  DictionaryAttr dictionary;
  OptionalParseResult parsed = parser.parseOptionalAttribute(dictionary);
  if (!parsed.has_value())
    return success();
  if (failed(*parsed))
    return failure();

  result.addAttributes(dictionary.getValue());
  return success();
}

static void printFunctionType(OpAsmPrinter &printer, FunctionType type) {
  printer << " (";
  if (type)
    llvm::interleaveComma(type.getInputs(), printer);
  printer << ") -> ";
  if (type && type.getNumResults() == 1) {
    printer << type.getResult(0);
    return;
  }
  printer << '(';
  if (type)
    llvm::interleaveComma(type.getResults(), printer);
  printer << ')';
}

static void readCountsForPrinting(ArrayAttr parameters, unsigned &loadCount,
                                  unsigned &storeCount) {
  loadCount = 0;
  storeCount = 0;
  if (!parameters || parameters.size() != 1)
    return;
  auto dictionary = dyn_cast<DictionaryAttr>(parameters[0]);
  if (!dictionary)
    return;
  if (auto attr = dyn_cast_or_null<IntegerAttr>(dictionary.get(kLoadGroupSize)))
    if (attr.getInt() >= 0)
      loadCount = static_cast<unsigned>(attr.getInt());
  if (auto attr =
          dyn_cast_or_null<IntegerAttr>(dictionary.get(kStoreGroupSize)))
    if (attr.getInt() >= 0)
      storeCount = static_cast<unsigned>(attr.getInt());
}

static SmallVector<int32_t> makeOrdinalRange(unsigned count) {
  SmallVector<int32_t> ordinals;
  ordinals.reserve(count);
  for (unsigned ordinal = 0; ordinal < count; ++ordinal)
    ordinals.push_back(static_cast<int32_t>(ordinal));
  return ordinals;
}

static unsigned countLeadingMemoryEndpoints(ArrayRef<Type> types) {
  unsigned count = 0;
  for (Type type : types) {
    if (!isa<MemRefType>(type))
      break;
    ++count;
  }
  return count;
}

static void addLegacyMemoryContract(MLIRContext *context,
                                    OperationState &result, Schedule schedule,
                                    ArrayRef<Type> inputs,
                                    ArrayRef<Type> results) {
  SmallVector<int32_t> managers =
      makeOrdinalRange(countLeadingMemoryEndpoints(inputs));
  SmallVector<int32_t> subordinates =
      makeOrdinalRange(countLeadingMemoryEndpoints(results));
  result.addAttribute(
      "memory_contract",
      MemoryContractAttr::get(context, MemoryEngineAttr::get(context, schedule),
                              LocalMemoryServiceAttr(),
                              DenseI32ArrayAttr::get(context, managers),
                              DenseI32ArrayAttr::get(context, subordinates)));
}

static ParseResult
parseMemoryContractPrefix(OpAsmParser &parser, OperationState &result,
                          std::optional<Schedule> &legacySchedule) {
  if (succeeded(parser.parseOptionalLSquare())) {
    StringRef scheduleKeyword;
    SMLoc scheduleLocation = parser.getCurrentLocation();
    if (parser.parseKeyword(&scheduleKeyword) || parser.parseRSquare())
      return failure();
    legacySchedule = symbolizeSchedule(scheduleKeyword);
    if (!legacySchedule)
      return parser.emitError(
                 scheduleLocation,
                 "expected fabric mem schedule keyword 'spatial' or "
                 "'temporal', got '")
             << scheduleKeyword << "'";
    return success();
  }

  if (parser.parseKeyword("contract"))
    return failure();
  MemoryContractAttr contract;
  if (parser.parseAttribute(contract))
    return failure();
  result.addAttribute("memory_contract", contract);
  return success();
}

static bool matchesOrdinalRange(DenseI32ArrayAttr endpoints, unsigned count) {
  if (!endpoints || endpoints.size() != count)
    return false;
  for (auto [ordinal, endpoint] : llvm::enumerate(endpoints.asArrayRef()))
    if (endpoint != static_cast<int32_t>(ordinal))
      return false;
  return true;
}

static bool canPrintLegacyEngineShorthand(MemOp op) {
  MemoryContractAttr contract = op.getMemoryContract();
  if (!contract || !contract.getEngine() || contract.getLocalService())
    return false;

  SmallVector<Type> inputs;
  SmallVector<Type> results;
  if (auto functionTypeAttr = op.getFunctionTypeAttr()) {
    auto functionType = dyn_cast<FunctionType>(functionTypeAttr.getValue());
    if (!functionType)
      return false;
    inputs.append(functionType.getInputs().begin(),
                  functionType.getInputs().end());
    results.append(functionType.getResults().begin(),
                   functionType.getResults().end());
  } else {
    for (Value input : op.getInputs())
      inputs.push_back(input.getType());
    results.append(op.getResultTypes().begin(), op.getResultTypes().end());
  }

  unsigned managerCount = countLeadingMemoryEndpoints(inputs);
  unsigned subordinateCount = countLeadingMemoryEndpoints(results);
  return matchesOrdinalRange(contract.getManagerEndpoints(), managerCount) &&
         matchesOrdinalRange(contract.getSubordinateEndpoints(),
                             subordinateCount);
}

static LogicalResult
collectAnonymousInputPortTypes(MemOp op,
                               SmallVectorImpl<Type> &inputPortTypes) {
  ArrayRef<Type> innerTypes = op.getInnerInputTypes();
  if (!innerTypes.empty()) {
    inputPortTypes.append(innerTypes.begin(), innerTypes.end());
  } else {
    for (Value input : op.getInputs())
      inputPortTypes.push_back(input.getType());
  }

  for (auto [index, pair] :
       llvm::enumerate(llvm::zip(op.getInputs(), inputPortTypes))) {
    Value input;
    Type inputPortType;
    std::tie(input, inputPortType) = pair;
    Type sourceType = input.getType();
    if (isa<MemRefType>(sourceType) || isa<MemRefType>(inputPortType)) {
      if (sourceType != inputPortType)
        return op.emitOpError("incoming connection operand #")
               << index
               << ": memref capabilities cannot use the 'to "
                  "<destination-type>' clause; memref types must match "
                  "exactly";
      continue;
    }
    if (!haveSameFabricModulePortKind(sourceType, inputPortType))
      return op.emitOpError("incoming connection operand #")
             << index << " source type " << sourceType
             << " and destination port type " << inputPortType
             << " must share the same fabric kind (bits or bits_tag)";
  }
  return success();
}

static LogicalResult readHardwareParameters(MemOp op,
                                            DictionaryAttr &dictionary) {
  ArrayAttr parameters = op.getHwParamsAttr();
  if (!parameters)
    return op.emitOpError(
        "requires 'hw_params' with operation-engine hardware parameters");
  if (parameters.size() != 1)
    return op.emitOpError(
               "'hw_params' must be a length-1 array wrapping a dictionary, "
               "got length ")
           << parameters.size();
  dictionary = dyn_cast<DictionaryAttr>(parameters[0]);
  if (!dictionary)
    return op.emitOpError("'hw_params' inner element must be a dictionary");
  return success();
}

static LogicalResult readI32(MemOp op, DictionaryAttr dictionary, StringRef key,
                             int64_t minimum, unsigned &value) {
  Attribute raw = dictionary.get(key);
  if (!raw)
    return op.emitOpError("'hw_params' missing required key '") << key << "'";
  auto attr = dyn_cast<IntegerAttr>(raw);
  auto type = attr ? dyn_cast<IntegerType>(attr.getType()) : IntegerType{};
  if (!attr || !type || !type.isSignless() || type.getWidth() != 32)
    return op.emitOpError("'hw_params' key '")
           << key << "' must be a signless i32";
  int64_t signedValue = attr.getInt();
  if (signedValue < minimum)
    return op.emitOpError("'hw_params' key '")
           << key << "' must be >= " << minimum << ", got " << signedValue;
  value = static_cast<unsigned>(signedValue);
  return success();
}

static LogicalResult verifyDictionaryKeys(MemOp op, DictionaryAttr dictionary,
                                          StringRef dictionaryName,
                                          ArrayRef<StringRef> allowed) {
  for (NamedAttribute field : dictionary) {
    StringRef name = field.getName().getValue();
    if (llvm::is_contained(allowed, name))
      continue;
    return op.emitOpError(dictionaryName)
           << " contains unsupported key '" << name << "'";
  }
  return success();
}

static LogicalResult verifyOperationEngine(MemOp op, DictionaryAttr dictionary,
                                           Schedule schedule,
                                           EngineInfo &engine) {
  if (dictionary.get(kLocalMemoryService))
    return op.emitOpError("'hw_params' key '")
           << kLocalMemoryService
           << "' describes the confirmed optional LocalMemoryService, which "
              "must be represented by the typed memory_contract";

  bool temporal = schedule == Schedule::Temporal;
  if (!temporal) {
    for (StringRef key : {StringRef(kTagWidth), StringRef(kOperationTableSize)})
      if (dictionary.get(key))
        return op.emitOpError("spatial fabric.mem must not carry "
                              "temporal-only key '")
               << key << "'";
  }

  const StringRef spatialKeys[] = {kLoadGroupSize, kStoreGroupSize, kDataWidth,
                                   kDispatchEligibility};
  const StringRef temporalKeys[] = {kLoadGroupSize,      kStoreGroupSize,
                                    kDataWidth,          kTagWidth,
                                    kOperationTableSize, kDispatchEligibility};
  if (failed(verifyDictionaryKeys(op, dictionary, "'hw_params'",
                                  temporal ? ArrayRef<StringRef>(temporalKeys)
                                           : ArrayRef<StringRef>(spatialKeys))))
    return failure();

  if (failed(readI32(op, dictionary, kLoadGroupSize, 0, engine.loadCount)) ||
      failed(readI32(op, dictionary, kStoreGroupSize, 0, engine.storeCount)) ||
      failed(readI32(op, dictionary, kDataWidth, 1, engine.dataWidth)))
    return failure();
  if (engine.loadCount + engine.storeCount == 0)
    return op.emitOpError(
        "load_group_size + store_group_size must be greater than zero");

  if (!temporal)
    return success();
  if (failed(readI32(op, dictionary, kTagWidth, 1, engine.tagWidth)) ||
      failed(readI32(op, dictionary, kOperationTableSize, 1,
                     engine.operationTableSize)))
    return failure();
  return success();
}

static LogicalResult verifyTemporalResidentCapacity(MemOp op, Schedule schedule,
                                                    const EngineInfo &engine) {
  if (schedule != Schedule::Temporal)
    return success();

  uint64_t physicalPortCount =
      static_cast<uint64_t>(engine.loadCount) + engine.storeCount;
  uint64_t representableRows = std::numeric_limits<uint64_t>::max();
  if (engine.tagWidth < std::numeric_limits<uint64_t>::digits) {
    uint64_t tagCount = uint64_t{1} << engine.tagWidth;
    if (physicalPortCount <= std::numeric_limits<uint64_t>::max() / tagCount)
      representableRows = physicalPortCount * tagCount;
  }
  if (engine.operationTableSize > representableRows)
    return op.emitOpError("operation_table_size ")
           << engine.operationTableSize
           << " exceeds representable temporal row capacity "
           << representableRows;
  return success();
}

static LogicalResult deriveSignatureLayout(MemOp op, ArrayRef<Type> inputs,
                                           ArrayRef<Type> results,
                                           const EngineInfo &engine,
                                           SignatureLayout &layout) {
  uint64_t operationInputs = 2ull * engine.loadCount + 3ull * engine.storeCount;
  uint64_t operationResults =
      2ull * engine.loadCount + static_cast<uint64_t>(engine.storeCount);
  if (inputs.size() < operationInputs)
    return op.emitOpError("signature has ")
           << inputs.size() << " input types but the operation engine requires "
           << operationInputs << " operation input types";
  if (results.size() < operationResults)
    return op.emitOpError("signature has ")
           << results.size()
           << " result types but the operation engine requires "
           << operationResults << " operation result types";

  layout.managerCount = static_cast<unsigned>(inputs.size() - operationInputs);
  layout.subordinateCount =
      static_cast<unsigned>(results.size() - operationResults);
  layout.loadOperandBase = layout.managerCount;
  layout.storeOperandBase = layout.loadOperandBase + 2 * engine.loadCount;
  layout.loadResultBase = layout.subordinateCount;
  layout.storeResultBase = layout.loadResultBase + 2 * engine.loadCount;
  return success();
}

static LogicalResult verifyCapabilityEndpoint(MemOp op, Type type,
                                              StringRef role, unsigned index) {
  auto memref = dyn_cast<MemRefType>(type);
  if (!memref)
    return op.emitOpError(role)
           << " endpoint #" << index << " must be a memref type, got " << type;
  if (!isa<BitsType>(memref.getElementType()))
    return op.emitOpError(role)
           << " endpoint #" << index
           << " element type must be '!fabric.bits<W>', got "
           << memref.getElementType();
  return success();
}

static LogicalResult verifyEndpointReferences(MemOp op, ArrayRef<Type> inputs,
                                              ArrayRef<Type> results,
                                              MemoryContractAttr contract) {
  auto verifyRole = [&](DenseI32ArrayAttr endpoints, ArrayRef<Type> types,
                        StringRef role) -> LogicalResult {
    SmallVector<bool> classified(types.size(), false);
    for (int32_t endpoint : endpoints.asArrayRef()) {
      if (endpoint < 0 || static_cast<uint64_t>(endpoint) >= types.size())
        return op.emitOpError(role)
               << " endpoint ordinal " << endpoint << " is outside [0, "
               << types.size() << ')';
      unsigned ordinal = static_cast<unsigned>(endpoint);
      if (failed(verifyCapabilityEndpoint(op, types[ordinal], role, ordinal)))
        return failure();
      classified[ordinal] = true;
    }

    for (auto [ordinal, type] : llvm::enumerate(types)) {
      if (isa<MemRefType>(type) && !classified[ordinal])
        return op.emitOpError("memory_contract does not classify ")
               << role << " endpoint at " << role << " signature ordinal "
               << ordinal;
    }
    return success();
  };

  if (failed(verifyRole(contract.getManagerEndpoints(), inputs, "manager")) ||
      failed(verifyRole(contract.getSubordinateEndpoints(), results,
                        "subordinate")))
    return failure();
  return success();
}

static LogicalResult verifyEngineEndpointLayout(MemOp op,
                                                MemoryContractAttr contract,
                                                const SignatureLayout &layout) {
  if (!matchesOrdinalRange(contract.getManagerEndpoints(), layout.managerCount))
    return op.emitOpError(
        "operation engine manager endpoints must be the leading input "
        "positions implied by its port parameters");
  if (!matchesOrdinalRange(contract.getSubordinateEndpoints(),
                           layout.subordinateCount))
    return op.emitOpError(
        "operation engine subordinate endpoints must be the leading result "
        "positions implied by its port parameters");
  return success();
}

static LogicalResult verifyOperationPortTypes(MemOp op, ArrayRef<Type> inputs,
                                              ArrayRef<Type> results,
                                              Schedule schedule,
                                              const EngineInfo &engine,
                                              const SignatureLayout &layout) {
  bool temporal = schedule == Schedule::Temporal;
  auto makePortType = [&](unsigned width) -> Type {
    if (temporal)
      return BitsTagType::get(op.getContext(), width, engine.tagWidth);
    return BitsType::get(op.getContext(), width);
  };
  Type expectedAddress = makePortType(::loom::getIndexWidth());
  Type expectedData = makePortType(engine.dataWidth);
  Type expectedControl = makePortType(0);

  auto checkType = [&](Type actual, Type expected, StringRef role,
                       unsigned index) -> LogicalResult {
    if (actual == expected)
      return success();
    return op.emitOpError(role) << " #" << index << " must have type "
                                << expected << ", got " << actual;
  };

  for (unsigned index = 0; index < engine.loadCount; ++index) {
    unsigned base = layout.loadOperandBase + 2 * index;
    if (failed(checkType(inputs[base], expectedAddress, "load address port",
                         index)) ||
        failed(checkType(inputs[base + 1], expectedControl, "load control port",
                         index)))
      return failure();
  }

  for (unsigned index = 0; index < engine.storeCount; ++index) {
    unsigned base = layout.storeOperandBase + 3 * index;
    if (failed(checkType(inputs[base], expectedAddress, "store address port",
                         index)))
      return failure();
    if (inputs[base + 1] != expectedData)
      return op.emitOpError("store data port #")
             << index << " must have operation data width " << engine.dataWidth
             << ", got " << inputs[base + 1];
    if (failed(checkType(inputs[base + 2], expectedControl,
                         "store control port", index)))
      return failure();
  }

  for (unsigned index = 0; index < engine.loadCount; ++index) {
    unsigned base = layout.loadResultBase + 2 * index;
    if (results[base] != expectedData)
      return op.emitOpError("load data port #")
             << index << " must have operation data width " << engine.dataWidth
             << ", got " << results[base];
    if (failed(checkType(results[base + 1], expectedControl,
                         "load completion port", index)))
      return failure();
  }

  for (unsigned index = 0; index < engine.storeCount; ++index)
    if (failed(checkType(results[layout.storeResultBase + index],
                         expectedControl, "store completion port", index)))
      return failure();
  return success();
}

static LogicalResult verifyDispatchEligibility(MemOp op,
                                               DictionaryAttr dictionary,
                                               const EngineInfo &engine,
                                               const SignatureLayout &layout) {
  auto eligibility =
      dyn_cast_or_null<DictionaryAttr>(dictionary.get(kDispatchEligibility));
  if (!eligibility)
    return op.emitOpError(
        "'hw_params' key 'dispatch_eligibility' must be a dictionary");

  const StringRef eligibilityKeys[] = {kOperationPortRequests,
                                       kSubordinateRequests};
  if (failed(verifyDictionaryKeys(op, eligibility, "dispatch_eligibility",
                                  eligibilityKeys)))
    return failure();

  auto verifyDomains = [&](StringRef key, unsigned sourceCount,
                           StringRef sourceCountName) -> LogicalResult {
    auto domains = dyn_cast_or_null<ArrayAttr>(eligibility.get(key));
    if (!domains)
      return op.emitOpError("dispatch_eligibility key '")
             << key << "' must be an array";
    if (domains.size() != sourceCount)
      return op.emitOpError(key)
             << " length " << domains.size() << " must equal "
             << sourceCountName << ' ' << sourceCount;

    for (auto [source, entry] : llvm::enumerate(domains)) {
      auto domain = dyn_cast<ArrayAttr>(entry);
      if (!domain)
        return op.emitOpError(key)
               << " entry #" << source << " must be an array";
      if (domain.empty())
        return op.emitOpError(key)
               << " entry #" << source << " must be non-empty";

      int64_t previous = -1;
      for (Attribute rawTarget : domain) {
        auto target = dyn_cast<IntegerAttr>(rawTarget);
        auto type =
            target ? dyn_cast<IntegerType>(target.getType()) : IntegerType{};
        if (!target || !type || !type.isSignless() || type.getWidth() != 32)
          return op.emitOpError(key)
                 << " entry #" << source
                 << " manager target identities must be signless i32 values";
        int64_t value = target.getInt();
        if (value < 0 || static_cast<uint64_t>(value) >= layout.managerCount)
          return op.emitOpError(key)
                 << " entry #" << source << " manager target identity " << value
                 << " is outside [0, " << layout.managerCount << ")";
        if (value <= previous)
          return op.emitOpError(key)
                 << " entry #" << source << " must be strictly increasing";
        previous = value;
      }
    }
    return success();
  };

  unsigned physicalPortCount = engine.loadCount + engine.storeCount;
  if (failed(verifyDomains(kOperationPortRequests, physicalPortCount,
                           "physical operation port count")) ||
      failed(verifyDomains(kSubordinateRequests, layout.subordinateCount,
                           "subordinate endpoint count")))
    return failure();
  return success();
}

static LogicalResult verifyCanonicalAttributeSet(MemOp op) {
  for (NamedAttribute attribute : op->getDiscardableAttrs())
    return op.emitOpError("has non-canonical discardable attribute '")
           << attribute.getName().getValue() << "'";
  return success();
}

} // namespace

ParseResult MemOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr name;
  bool named = succeeded(parser.parseOptionalSymbolName(
      name, SymbolTable::getSymbolAttrName(), result.attributes));

  std::optional<Schedule> legacySchedule;
  if (parseMemoryContractPrefix(parser, result, legacySchedule))
    return failure();

  if (named) {
    if (parseFunctionType(parser, result) ||
        parseHardwareParameters(parser, result) ||
        parseDiscardableAttributes(parser, result))
      return failure();
    if (legacySchedule) {
      auto typeAttr =
          dyn_cast<TypeAttr>(result.attributes.get("function_type"));
      auto functionType = typeAttr ? dyn_cast<FunctionType>(typeAttr.getValue())
                                   : FunctionType{};
      if (!functionType)
        return failure();
      addLegacyMemoryContract(parser.getContext(), result, *legacySchedule,
                              functionType.getInputs(),
                              functionType.getResults());
    }
    return success();
  }

  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SMLoc operandsLocation = parser.getCurrentLocation();
  if (parser.parseKeyword("mgr"))
    return failure();
  SmallVector<OpAsmParser::UnresolvedOperand> managers;
  if (parser.parseOperandList(managers, OpAsmParser::Delimiter::Paren))
    return failure();
  operands.append(managers);

  if (succeeded(parser.parseOptionalKeyword("load"))) {
    SmallVector<OpAsmParser::UnresolvedOperand> loadOperands;
    if (parser.parseOperandList(loadOperands, OpAsmParser::Delimiter::Paren))
      return failure();
    operands.append(loadOperands);
  }

  if (succeeded(parser.parseOptionalKeyword("store"))) {
    SmallVector<OpAsmParser::UnresolvedOperand> storeOperands;
    if (parser.parseOperandList(storeOperands, OpAsmParser::Delimiter::Paren))
      return failure();
    operands.append(storeOperands);
  }

  if (parseHardwareParameters(parser, result) ||
      parseDiscardableAttributes(parser, result) || parser.parseColon())
    return failure();

  SmallVector<Type> sourceTypes;
  SmallVector<Type> inputPortTypes;
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    auto parseInputType = [&]() -> ParseResult {
      Type sourceType;
      if (parser.parseType(sourceType))
        return failure();
      Type inputPortType = sourceType;
      if (succeeded(parser.parseOptionalKeyword("to")) &&
          parser.parseType(inputPortType))
        return failure();
      sourceTypes.push_back(sourceType);
      inputPortTypes.push_back(inputPortType);
      return success();
    };
    if (parseInputType())
      return failure();
    while (succeeded(parser.parseOptionalComma()))
      if (parseInputType())
        return failure();
    if (parser.parseRParen())
      return failure();
  }

  if (parser.parseArrow())
    return failure();
  SmallVector<Type> resultTypes;
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parser.parseOptionalRParen())) {
      if (parser.parseTypeList(resultTypes) || parser.parseRParen())
        return failure();
    }
  } else {
    Type type;
    if (parser.parseType(type))
      return failure();
    resultTypes.push_back(type);
  }

  if (sourceTypes.size() != operands.size())
    return parser.emitError(operandsLocation,
                            "operand count does not match type list count");
  if (parser.resolveOperands(operands, sourceTypes, operandsLocation,
                             result.operands))
    return failure();

  if (!llvm::equal(sourceTypes, inputPortTypes))
    result.getOrAddProperties<Properties>().setInnerInputTypes(inputPortTypes);
  result.addTypes(resultTypes);
  if (legacySchedule)
    addLegacyMemoryContract(parser.getContext(), result, *legacySchedule,
                            inputPortTypes, resultTypes);
  return success();
}

void MemOp::print(OpAsmPrinter &printer) {
  bool named = static_cast<bool>(getSymNameAttr());
  if (named) {
    printer << ' ';
    printer.printSymbolName(getSymNameAttr().getValue());
  }

  unsigned loadCount = 0;
  unsigned storeCount = 0;
  readCountsForPrinting(getHwParamsAttr(), loadCount, storeCount);
  if (canPrintLegacyEngineShorthand(*this)) {
    printer << " ["
            << stringifySchedule(getMemoryContract().getEngine().getSchedule())
            << "]";
  } else {
    printer << " contract ";
    printer.printAttribute(getMemoryContract());
  }

  if (named) {
    FunctionType type;
    if (auto attr = getFunctionTypeAttr())
      type = dyn_cast<FunctionType>(attr.getValue());
    printFunctionType(printer, type);
  } else {
    uint64_t operationInputs = 2ull * loadCount + 3ull * storeCount;
    unsigned managerCount =
        getInputs().size() >= operationInputs
            ? static_cast<unsigned>(getInputs().size() - operationInputs)
            : 0;
    printer << " mgr(";
    llvm::interleaveComma(getInputs().take_front(managerCount), printer);
    printer << ')';

    if (loadCount) {
      printer << " load(";
      llvm::interleaveComma(getInputs().slice(managerCount, 2 * loadCount),
                            printer);
      printer << ')';
    }
    if (storeCount) {
      printer << " store(";
      llvm::interleaveComma(
          getInputs().slice(managerCount + 2 * loadCount, 3 * storeCount),
          printer);
      printer << ')';
    }
  }

  if (ArrayAttr parameters = getHwParamsAttr()) {
    printer << " [";
    llvm::interleaveComma(parameters, printer, [&](Attribute attr) {
      printer.printAttribute(attr);
    });
    printer << ']';
  }

  SmallVector<NamedAttribute> discardableAttributes =
      llvm::to_vector(getOperation()->getDiscardableAttrs());
  printer.printOptionalAttrDict(discardableAttributes);

  if (named)
    return;

  ArrayRef<Type> innerTypes = getInnerInputTypes();
  SmallVector<Type> inputPortTypes;
  if (!innerTypes.empty() && innerTypes.size() == getInputs().size()) {
    inputPortTypes.append(innerTypes.begin(), innerTypes.end());
  } else {
    for (Value input : getInputs())
      inputPortTypes.push_back(input.getType());
  }

  printer << " : (";
  llvm::interleaveComma(llvm::zip(getInputs(), inputPortTypes), printer,
                        [&](auto pair) {
                          Value input;
                          Type inputPortType;
                          std::tie(input, inputPortType) = pair;
                          printer << input.getType();
                          if (inputPortType != input.getType())
                            printer << " to " << inputPortType;
                        });
  printer << ") -> ";
  if (getNumResults() == 1) {
    printer << getResultTypes().front();
  } else {
    printer << '(';
    llvm::interleaveComma(getResultTypes(), printer);
    printer << ')';
  }
}

bool MemOp::isOptionalSymbol() { return true; }

LogicalResult MemOp::verify() {
  if (failed(verifyCanonicalAttributeSet(*this)))
    return failure();
  if (failed(verifyInnerInputTypesProperty(getOperation(), getInputs(),
                                           getInnerInputTypes())))
    return failure();

  bool named = static_cast<bool>(getSymNameAttr());
  SmallVector<Type> inputs;
  SmallVector<Type> results;
  if (named) {
    if (!getInputs().empty())
      return emitOpError(
                 "named fabric.mem template must have zero SSA operands; got ")
             << getInputs().size();
    if (getNumResults())
      return emitOpError(
                 "named fabric.mem template must have zero SSA results; got ")
             << getNumResults();
    if (!getInnerInputTypes().empty())
      return emitOpError("named fabric.mem template must not carry '")
             << kInnerInputTypesPropertyName << "'";
    auto attr = getFunctionTypeAttr();
    if (!attr)
      return emitOpError(
          "named fabric.mem template requires a 'function_type' attribute");
    auto type = dyn_cast<FunctionType>(attr.getValue());
    if (!type)
      return emitOpError("'function_type' must be a FunctionType");
    inputs.append(type.getInputs().begin(), type.getInputs().end());
    results.append(type.getResults().begin(), type.getResults().end());
  } else {
    if (getFunctionTypeAttr())
      return emitOpError(
          "anonymous fabric.mem must not carry a 'function_type' attribute");
    if (failed(collectAnonymousInputPortTypes(*this, inputs)))
      return failure();
    results.append(getResultTypes().begin(), getResultTypes().end());
  }

  MemoryContractAttr contract = getMemoryContract();
  if (!contract)
    return emitOpError("requires a typed memory_contract");
  if (failed(verifyEndpointReferences(*this, inputs, results, contract)))
    return failure();

  MemoryEngineAttr engineContract = contract.getEngine();
  LocalMemoryServiceAttr localService = contract.getLocalService();
  ArrayRef<int32_t> managerEndpoints =
      contract.getManagerEndpoints().asArrayRef();
  ArrayRef<int32_t> subordinateEndpoints =
      contract.getSubordinateEndpoints().asArrayRef();

  if (!engineContract && !localService && !subordinateEndpoints.empty())
    return emitOpError(
        "subordinate endpoint requires an Operation Engine or Local Memory "
        "Service");
  if (!engineContract && !managerEndpoints.empty())
    return emitOpError("manager endpoint requires an Operation Engine");
  if (!engineContract && !localService)
    return emitOpError("requires an Operation Engine or Local Memory Service");
  if (!engineContract) {
    if (subordinateEndpoints.empty())
      return emitOpError(
          "storage-only occurrence requires at least one subordinate "
          "endpoint");
    if (!inputs.empty())
      return emitOpError("storage-only occurrence must have zero input ports");
    if (results.size() != subordinateEndpoints.size())
      return emitOpError(
          "storage-only occurrence results must exactly match the "
          "subordinate endpoint inventory");
    if (getHwParamsAttr())
      return emitOpError(
          "storage-only occurrence must not carry operation-engine "
          "hw_params");
    return success();
  }
  if (!localService && managerEndpoints.empty())
    return emitOpError(
        "operation-engine-only occurrence requires at least one manager "
        "endpoint");

  Schedule schedule = engineContract.getSchedule();
  DictionaryAttr hardware;
  if (failed(readHardwareParameters(*this, hardware)))
    return failure();
  EngineInfo engine;
  if (failed(verifyOperationEngine(*this, hardware, schedule, engine)))
    return failure();
  if (failed(verifyTemporalResidentCapacity(*this, schedule, engine)))
    return failure();

  SignatureLayout layout;
  if (failed(deriveSignatureLayout(*this, inputs, results, engine, layout)))
    return failure();
  if (failed(verifyEngineEndpointLayout(*this, contract, layout)))
    return failure();

  if (failed(verifyOperationPortTypes(*this, inputs, results, schedule, engine,
                                      layout)))
    return failure();
  if ((hardware.get(kDispatchEligibility) || !localService) &&
      failed(verifyDispatchEligibility(*this, hardware, engine, layout)))
    return failure();
  return success();
}
