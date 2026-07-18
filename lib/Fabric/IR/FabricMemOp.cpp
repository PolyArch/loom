//===- FabricMemOp.cpp - Parser/printer/verifier for fabric.mem -----------===//
//
// Implements the operation-engine hardware capability ABI for fabric.mem.
// Manager and subordinate endpoint counts are derived from the signature.
// The hardware dictionary owns independent L, S, and W parameters. Temporal
// engines additionally own T, K, and fixed slot-to-physical-port eligibility.
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

static LogicalResult verifyHardwareKeys(MemOp op, DictionaryAttr dictionary,
                                        ArrayRef<StringRef> allowed) {
  for (NamedAttribute field : dictionary) {
    StringRef name = field.getName().getValue();
    if (llvm::is_contained(allowed, name))
      continue;
    return op.emitOpError("'hw_params' contains unsupported key '")
           << name << "'";
  }
  return success();
}

static LogicalResult verifyOperationEngine(MemOp op, DictionaryAttr dictionary,
                                           EngineInfo &engine) {
  bool temporal = op.getSchedule() == Schedule::Temporal;
  if (!temporal) {
    for (StringRef key : {StringRef(kTagWidth), StringRef(kOperationTableSize),
                          StringRef(kDispatchEligibility)})
      if (dictionary.get(key))
        return op.emitOpError("spatial fabric.mem must not carry "
                              "temporal-only key '")
               << key << "'";
  }

  const StringRef spatialKeys[] = {kLoadGroupSize, kStoreGroupSize, kDataWidth};
  const StringRef temporalKeys[] = {kLoadGroupSize,      kStoreGroupSize,
                                    kDataWidth,          kTagWidth,
                                    kOperationTableSize, kDispatchEligibility};
  if (failed(verifyHardwareKeys(op, dictionary,
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

static LogicalResult verifyOperationPortTypes(MemOp op, ArrayRef<Type> inputs,
                                              ArrayRef<Type> results,
                                              const EngineInfo &engine,
                                              const SignatureLayout &layout) {
  bool temporal = op.getSchedule() == Schedule::Temporal;
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
                                               const EngineInfo &engine) {
  auto eligibility =
      dyn_cast_or_null<ArrayAttr>(dictionary.get(kDispatchEligibility));
  if (!eligibility)
    return op.emitOpError(
        "'hw_params' key 'dispatch_eligibility' must be an array");
  if (eligibility.size() != engine.operationTableSize)
    return op.emitOpError("dispatch_eligibility length ")
           << eligibility.size() << " must equal operation_table_size "
           << engine.operationTableSize;

  unsigned physicalPortCount = engine.loadCount + engine.storeCount;
  for (auto [slot, entry] : llvm::enumerate(eligibility)) {
    auto domain = dyn_cast<ArrayAttr>(entry);
    if (!domain)
      return op.emitOpError("dispatch_eligibility entry #")
             << slot << " must be an array";
    if (domain.empty())
      return op.emitOpError("dispatch_eligibility entry #")
             << slot << " must be non-empty";

    int64_t previous = -1;
    for (Attribute rawPort : domain) {
      auto port = dyn_cast<IntegerAttr>(rawPort);
      auto type = port ? dyn_cast<IntegerType>(port.getType()) : IntegerType{};
      if (!port || !type || !type.isSignless() || type.getWidth() != 32)
        return op.emitOpError("dispatch_eligibility entry #")
               << slot << " port identities must be signless i32 values";
      int64_t value = port.getInt();
      if (value < 0 || static_cast<uint64_t>(value) >= physicalPortCount)
        return op.emitOpError("dispatch_eligibility entry #")
               << slot << " port identity " << value << " is outside [0, "
               << physicalPortCount << ")";
      if (value <= previous)
        return op.emitOpError("dispatch_eligibility entry #")
               << slot << " must be strictly increasing";
      previous = value;
    }
  }
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

  StringRef scheduleKeyword;
  SMLoc scheduleLocation = parser.getCurrentLocation();
  if (parser.parseLSquare() || parser.parseKeyword(&scheduleKeyword) ||
      parser.parseRSquare())
    return failure();
  std::optional<Schedule> schedule = symbolizeSchedule(scheduleKeyword);
  if (!schedule)
    return parser.emitError(scheduleLocation,
                            "expected fabric mem schedule keyword 'spatial' or "
                            "'temporal', got '")
           << scheduleKeyword << "'";
  result.addAttribute("schedule",
                      ScheduleAttr::get(parser.getContext(), *schedule));

  if (named) {
    if (parseFunctionType(parser, result) ||
        parseHardwareParameters(parser, result) ||
        parseDiscardableAttributes(parser, result))
      return failure();
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
  return success();
}

void MemOp::print(OpAsmPrinter &printer) {
  bool named = static_cast<bool>(getSymNameAttr());
  if (named) {
    printer << ' ';
    printer.printSymbolName(getSymNameAttr().getValue());
  }
  printer << " [" << stringifySchedule(getSchedule()) << "]";

  unsigned loadCount = 0;
  unsigned storeCount = 0;
  readCountsForPrinting(getHwParamsAttr(), loadCount, storeCount);

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

  DictionaryAttr hardware;
  if (failed(readHardwareParameters(*this, hardware)))
    return failure();
  EngineInfo engine;
  if (failed(verifyOperationEngine(*this, hardware, engine)))
    return failure();

  SignatureLayout layout;
  if (failed(deriveSignatureLayout(*this, inputs, results, engine, layout)))
    return failure();
  if (layout.managerCount == 0)
    return emitOpError(
        "operation engine requires at least one manager endpoint");
  for (unsigned index = 0; index < layout.managerCount; ++index)
    if (failed(
            verifyCapabilityEndpoint(*this, inputs[index], "manager", index)))
      return failure();
  for (unsigned index = 0; index < layout.subordinateCount; ++index)
    if (failed(verifyCapabilityEndpoint(*this, results[index], "subordinate",
                                        index)))
      return failure();

  if (failed(verifyOperationPortTypes(*this, inputs, results, engine, layout)))
    return failure();
  if (getSchedule() == Schedule::Temporal &&
      failed(verifyDispatchEligibility(*this, hardware, engine)))
    return failure();
  return success();
}
