// Anchor aligns canonical ConfiguredFunctions from ordered outputs toward
// inputs. Compatible nodes share one fabric.op. Cross-share-group positions
// use explicit input demuxes and an output mux when enabled.

#include "Fabric/Tech/Synthesizer/Anchor.h"

#include "Common/HwShareGroup.h"
#include "Common/SynthConfig.h"
#include "Dataflow/IR/DataflowEnums.h"
#include "Fabric/IR/ConfiguredFunction.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/Tech/Synthesizer/CostModel.h"
#include "Fabric/Tech/Synthesizer/CoverageVerifier.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace loom::fabric::tech {

namespace {

//===----------------------------------------------------------------------===//
// Type lifting helpers.
//===----------------------------------------------------------------------===//
//
// Input subgraph block-arg types are software types (iN / fN / index /
// i1). The synthesized FU's signature exposes `fabric.bits<N>` ports
// only. Mirror the forward direction's bit-width assignments so the
// inverse lifting agrees with `SubgraphEnumerator::computePortLiftMap`
// and `liftFor`.

// Bit width of an MLIR software type expressible as a fabric.bits<N>
// port. Returns std::nullopt when the type cannot be lifted (caller
// treats as a topology mismatch). NoneType (e.g. dataflow.constant's
// ctrl input) lifts to bits<0> -- a legitimate, zero-width port.
::std::optional<unsigned> tryBitWidthOf(::mlir::Type t) {
  std::string error;
  auto width = ::fabric::getSemanticPayloadWidth(t, error);
  if (::mlir::failed(width))
    return std::nullopt;
  return *width;
}

::std::string printType(::mlir::Type type) {
  ::std::string text;
  ::llvm::raw_string_ostream os(text);
  type.print(os);
  return text;
}

::std::string printAttribute(::mlir::Attribute attribute) {
  ::std::string text;
  ::llvm::raw_string_ostream os(text);
  attribute.print(os);
  return text;
}

::mlir::Type cloneType(::mlir::Type type, ::mlir::MLIRContext *context) {
  return ::mlir::parseType(printType(type), context);
}

::mlir::Attribute cloneAttribute(::mlir::Attribute attribute,
                                 ::mlir::MLIRContext *context) {
  return ::mlir::parseAttribute(printAttribute(attribute), context);
}

::mlir::ArrayAttr indexArray(::mlir::MLIRContext *context, unsigned count) {
  auto i32 = ::mlir::IntegerType::get(context, 32);
  ::llvm::SmallVector<::mlir::Attribute, 4> values;
  for (unsigned index = 0; index < count; ++index)
    values.push_back(::mlir::IntegerAttr::get(i32, index));
  return ::mlir::ArrayAttr::get(context, values);
}

struct PendingAssignment {
  ::mlir::Operation *resource = nullptr;
  ::mlir::DictionaryAttr payload;
};

using EncodingFragment = ::llvm::SmallVector<PendingAssignment, 6>;

struct EncodingChoiceGroup {
  ::llvm::SmallVector<EncodingFragment, 4> choices;
};

::std::optional<PendingAssignment>
opModeAssignment(::fabric::OpOp resource,
                 const ::fabric::ConfiguredFunctionNode &source,
                 ::mlir::MLIRContext *context) {
  ::llvm::SmallVector<::mlir::Type, 4> inputs;
  for (::mlir::Type type : source.functionType.getInputs()) {
    ::mlir::Type cloned = cloneType(type, context);
    if (!cloned)
      return std::nullopt;
    inputs.push_back(cloned);
  }
  ::llvm::SmallVector<::mlir::Type, 2> results;
  for (::mlir::Type type : source.functionType.getResults()) {
    ::mlir::Type cloned = cloneType(type, context);
    if (!cloned)
      return std::nullopt;
    results.push_back(cloned);
  }
  auto attributes = ::mlir::dyn_cast_or_null<::mlir::DictionaryAttr>(
      cloneAttribute(source.attributes, context));
  if (!attributes)
    return std::nullopt;

  auto functionType = ::mlir::FunctionType::get(context, inputs, results);
  ::llvm::SmallVector<::mlir::NamedAttribute, 5> fields = {
      {::mlir::StringAttr::get(context, "op"),
       ::mlir::FlatSymbolRefAttr::get(context, source.operationName)},
      {::mlir::StringAttr::get(context, "function_type"),
       ::mlir::TypeAttr::get(functionType)},
      {::mlir::StringAttr::get(context, "input_ports"),
       indexArray(context, source.functionType.getNumInputs())},
      {::mlir::StringAttr::get(context, "output_ports"),
       indexArray(context, source.functionType.getNumResults())},
      {::mlir::StringAttr::get(context, "attributes"), attributes}};
  auto mode = ::mlir::DictionaryAttr::get(context, fields);

  ::llvm::SmallVector<::mlir::Attribute, 4> modes;
  if (auto existing = resource->getAttrOfType<::mlir::ArrayAttr>("hw_params"))
    modes.append(existing.begin(), existing.end());
  unsigned modeIndex = 0;
  auto found = ::llvm::find(modes, mode);
  if (found == modes.end()) {
    modeIndex = modes.size();
    modes.push_back(mode);
    resource->setAttr("hw_params", ::mlir::ArrayAttr::get(context, modes));
  } else {
    modeIndex = static_cast<unsigned>(found - modes.begin());
  }

  auto i32 = ::mlir::IntegerType::get(context, 32);
  ::llvm::SmallVector<::mlir::NamedAttribute, 1> selection = {
      {::mlir::StringAttr::get(context, "mode"),
       ::mlir::IntegerAttr::get(i32, modeIndex)}};
  return PendingAssignment{resource.getOperation(),
                           ::mlir::DictionaryAttr::get(context, selection)};
}

PendingAssignment routeAssignment(::mlir::Operation *resource, unsigned select,
                                  ::mlir::MLIRContext *context) {
  auto i32 = ::mlir::IntegerType::get(context, 32);
  ::llvm::SmallVector<::mlir::NamedAttribute, 1> fields = {
      {::mlir::StringAttr::get(context, "select"),
       ::mlir::IntegerAttr::get(i32, select)}};
  return {resource, ::mlir::DictionaryAttr::get(context, fields)};
}

bool sameFragment(const EncodingFragment &lhs, const EncodingFragment &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (const PendingAssignment &assignment : lhs) {
    auto found = ::llvm::find_if(rhs, [&](const PendingAssignment &other) {
      return assignment.resource == other.resource;
    });
    if (found == rhs.end() || found->payload != assignment.payload)
      return false;
  }
  return true;
}

void addUniqueChoice(EncodingChoiceGroup &group, EncodingFragment fragment) {
  if (::llvm::any_of(group.choices, [&](const EncodingFragment &existing) {
        return sameFragment(existing, fragment);
      }))
    return;
  group.choices.push_back(std::move(fragment));
}

//===----------------------------------------------------------------------===//
// Worker-side state for one anchor synthesis run.
//===----------------------------------------------------------------------===//

// Identity of one aligned value position across the input functions.
struct PeerKey {
  ::llvm::SmallVector<::fabric::ConfiguredValue, 4> peers;
  bool operator==(const PeerKey &o) const { return peers == o.peers; }
};

struct PeerKeyInfo {
  static PeerKey getEmptyKey() {
    PeerKey k;
    k.peers.push_back(::fabric::ConfiguredValue::input(~0u));
    return k;
  }
  static PeerKey getTombstoneKey() {
    PeerKey k;
    k.peers.push_back(::fabric::ConfiguredValue::input(~0u - 1));
    return k;
  }
  static unsigned getHashValue(const PeerKey &k) {
    ::llvm::hash_code h = ::llvm::hash_value(static_cast<unsigned>(0));
    for (const ::fabric::ConfiguredValue &value : k.peers)
      h = ::llvm::hash_combine(h, static_cast<unsigned>(value.kind),
                               value.index, value.result);
    return static_cast<unsigned>(h);
  }
  static bool isEqual(const PeerKey &a, const PeerKey &b) { return a == b; }
};

// A fabric.bits<N> Value already emitted for a peer set.
struct EmittedSlot {
  ::mlir::Value value;
};

using PeerVec = ::llvm::SmallVector<::fabric::ConfiguredValue, 4>;

// Helper: build a PeerKey from a PeerVec.
PeerKey keyOf(const PeerVec &v) {
  PeerKey k;
  k.peers.assign(v.begin(), v.end());
  return k;
}

// One physical operation is shared by aligned references to its different
// results.
PeerKey opKeyOf(const PeerVec &v) {
  PeerKey k;
  k.peers.reserve(v.size());
  for (const ::fabric::ConfiguredValue &value : v) {
    ::fabric::ConfiguredValue projected = value;
    if (value.kind == ::fabric::ConfiguredValue::Kind::NodeResult)
      projected.result = 0;
    k.peers.push_back(projected);
  }
  return k;
}

//===----------------------------------------------------------------------===//
// Wrapper-port assignment (block-arg identity).
//===----------------------------------------------------------------------===//
//
// All inputs share one DAG topology and one block-argument shape. The
// wrapper exposes one physical input per block-argument index, sized to the
// maximum software payload width observed at that position.

// Compute the wrapper's input ports: one entry per block-argument index of
// the input functions.
struct WrapperInputSlot {
  unsigned port;
  unsigned bitwidth;
};

::std::optional<::llvm::SmallVector<WrapperInputSlot, 4>>
collectWrapperInputSlots(
    ::llvm::ArrayRef<::fabric::ConfiguredFunction> functions) {
  ::llvm::SmallVector<WrapperInputSlot, 4> ports;
  if (functions.empty())
    return ports;
  const ::fabric::ConfiguredFunction &first = functions.front();
  unsigned na = first.inputs.size();
  // Every input function must have the same argument count.
  for (const ::fabric::ConfiguredFunction &function : functions)
    if (function.inputs.size() != na)
      return std::nullopt;
  ports.reserve(na);
  for (unsigned i = 0; i < na; ++i) {
    unsigned port = first.inputs[i].fuPort;
    auto firstBw = tryBitWidthOf(first.inputs[i].type);
    if (!firstBw.has_value())
      return std::nullopt;
    unsigned bw = *firstBw;
    for (const ::fabric::ConfiguredFunction &function : functions) {
      if (function.inputs[i].fuPort != port)
        return std::nullopt;
      auto other = tryBitWidthOf(function.inputs[i].type);
      if (!other)
        return std::nullopt;
      bw = std::max(bw, *other);
    }
    ports.push_back({port, bw});
  }
  return ports;
}

// Compute the maximum physical width required by each software result
// position. All inputs must agree on result arity.
::std::optional<::llvm::SmallVector<unsigned, 4>> collectWrapperOutputWidths(
    ::llvm::ArrayRef<::fabric::ConfiguredFunction> functions) {
  ::llvm::SmallVector<unsigned, 4> widths;
  if (functions.empty())
    return widths;
  const ::fabric::ConfiguredFunction &first = functions.front();
  unsigned ar = first.outputs.size();
  for (const ::fabric::ConfiguredFunction &function : functions)
    if (function.outputs.size() != ar)
      return std::nullopt;
  widths.reserve(ar);
  for (unsigned i = 0; i < ar; ++i) {
    unsigned port = first.outputs[i].fuPort;
    unsigned width = 0;
    for (const ::fabric::ConfiguredFunction &function : functions) {
      if (function.outputs[i].fuPort != port)
        return std::nullopt;
      auto bw = tryBitWidthOf(function.outputs[i].type);
      if (!bw.has_value())
        return std::nullopt;
      width = std::max(width, *bw);
    }
    widths.push_back(width);
  }
  return widths;
}

//===----------------------------------------------------------------------===//
// Configured-function graph helpers.
//===----------------------------------------------------------------------===//

bool peersUniformKind(const PeerVec &peers) {
  if (peers.empty())
    return false;
  auto kind = peers.front().kind;
  for (const ::fabric::ConfiguredValue &value : peers)
    if (value.kind != kind)
      return false;
  return true;
}

const ::fabric::ConfiguredFunctionNode *
nodeFor(const ::fabric::ConfiguredFunction &function,
        const ::fabric::ConfiguredValue &value) {
  if (value.kind != ::fabric::ConfiguredValue::Kind::NodeResult ||
      value.index >= function.nodes.size())
    return nullptr;
  return &function.nodes[value.index];
}

::std::optional<unsigned>
widthOfSource(const ::fabric::ConfiguredValue &source,
              const ::fabric::ConfiguredFunction &function) {
  if (source.kind == ::fabric::ConfiguredValue::Kind::InputPort) {
    auto input = ::llvm::find_if(
        function.inputs, [&](const ::fabric::ConfiguredBoundaryInput &other) {
          return other.fuPort == source.index;
        });
    if (input == function.inputs.end())
      return std::nullopt;
    return tryBitWidthOf(input->type);
  }
  const auto *node = nodeFor(function, source);
  if (!node || source.result >= node->functionType.getNumResults())
    return std::nullopt;
  return tryBitWidthOf(node->functionType.getResult(source.result));
}

// Compute the physical width required by one peer value position.
bool collectPeerWidth(const PeerVec &peers,
                      ::llvm::ArrayRef<::fabric::ConfiguredFunction> functions,
                      unsigned &widthOut) {
  unsigned w = 0;
  for (auto [i, source] : ::llvm::enumerate(peers)) {
    auto cur = widthOfSource(source, functions[i]);
    if (!cur.has_value())
      return false;
    w = std::max(w, *cur);
  }
  widthOut = w;
  return true;
}

bool peersUniformArity(const PeerVec &peers,
                       ::llvm::ArrayRef<::fabric::ConfiguredFunction> functions,
                       unsigned &arityOut) {
  if (peers.empty())
    return false;
  const auto *first = nodeFor(functions.front(), peers.front());
  if (!first)
    return false;
  unsigned arity = first->operands.size();
  for (auto [i, source] : ::llvm::enumerate(peers)) {
    const auto *node = nodeFor(functions[i], source);
    if (!node || node->operands.size() != arity)
      return false;
    if (source.result != peers.front().result)
      return false;
  }
  arityOut = arity;
  return true;
}

bool peersShareStreamStepKind(
    const PeerVec &peers,
    ::llvm::ArrayRef<::fabric::ConfiguredFunction> functions) {
  ::std::optional<::dataflow::StreamStepKind> fixedStep;
  for (auto [index, source] : ::llvm::enumerate(peers)) {
    const auto *node = nodeFor(functions[index], source);
    if (!node || node->operationName != "dataflow.stream")
      continue;
    auto step = ::dataflow::getStreamStepKindFromAttr(
        node->attributes.get("step_kind"));
    if (!step)
      return false;
    if (fixedStep && *fixedStep != *step)
      return false;
    fixedStep = *step;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// fabric.op emission.
//===----------------------------------------------------------------------===//

// Build the sorted-union op_list ArrayAttr for a single share group.
// Sort key is the op-name string; the spec requires lexical order so
// the canonical printed form is deterministic.
::mlir::ArrayAttr sortedOpListFor(const ::std::set<::std::string> &names,
                                  ::mlir::MLIRContext *ctx) {
  ::llvm::SmallVector<::mlir::Attribute, 4> attrs;
  attrs.reserve(names.size());
  for (const ::std::string &n : names)
    attrs.push_back(::mlir::FlatSymbolRefAttr::get(ctx, n));
  return ::mlir::ArrayAttr::get(ctx, attrs);
}

// Construct one `fabric.op` instance with the given op_list, hw_params,
// inputs, and result types. The op is emitted at the builder's current
// insertion point.
::fabric::OpOp emitFabricOp(::mlir::OpBuilder &builder, ::mlir::Location loc,
                            ::mlir::ArrayAttr opList,
                            ::mlir::ValueRange operands,
                            ::mlir::TypeRange resultTypes) {
  ::mlir::OperationState state(loc, ::fabric::OpOp::getOperationName());
  state.addOperands(operands);
  state.addTypes(resultTypes);
  state.addAttribute("op_list", opList);
  ::mlir::Operation *raw = builder.create(state);
  return ::mlir::cast<::fabric::OpOp>(raw);
}

// Construct one `fabric.mux` over `arms` with the shared bits type.
::fabric::MuxOp emitFabricMux(::mlir::OpBuilder &builder, ::mlir::Location loc,
                              ::mlir::ValueRange arms, ::mlir::Type bits) {
  ::mlir::OperationState state(loc, ::fabric::MuxOp::getOperationName());
  state.addOperands(arms);
  state.addTypes({bits});
  ::mlir::Operation *raw = builder.create(state);
  return ::mlir::cast<::fabric::MuxOp>(raw);
}

::fabric::DemuxOp emitFabricDemux(::mlir::OpBuilder &builder,
                                  ::mlir::Location loc, ::mlir::Value input,
                                  unsigned outputCount) {
  ::mlir::OperationState state(loc, ::fabric::DemuxOp::getOperationName());
  state.addOperands(input);
  state.addTypes(
      ::llvm::SmallVector<::mlir::Type, 4>(outputCount, input.getType()));
  return ::mlir::cast<::fabric::DemuxOp>(builder.create(state));
}

//===----------------------------------------------------------------------===//
// Per-position decision: single-share-group vs cross-share-group merge.
//===----------------------------------------------------------------------===//

// Key for a share-group bucket. A multi-member group is identified by
// its index in `loom::common::hwShareGroups()`; a singleton op (any op
// not in any multi-member group) is identified by its own op name so
// that distinct singletons at the same anchor position never collapse
// into one bucket. Bucket order is deterministic: multi-member groups
// (sorted by index) come first, then singletons sorted by op name.
struct BucketKey {
  bool isSingleton = false;
  ::std::size_t groupIndex = 0; // valid iff !isSingleton
  ::std::string singletonName;  // valid iff isSingleton

  static BucketKey forGroup(::std::size_t idx) {
    BucketKey k;
    k.isSingleton = false;
    k.groupIndex = idx;
    return k;
  }
  static BucketKey forSingleton(::llvm::StringRef name) {
    BucketKey k;
    k.isSingleton = true;
    k.singletonName = name.str();
    return k;
  }
  static BucketKey forName(::llvm::StringRef name) {
    if (auto idx = ::loom::common::findShareGroup(name))
      return forGroup(*idx);
    return forSingleton(name);
  }

  bool operator<(const BucketKey &o) const {
    // Multi-member groups (isSingleton == false) sort before singletons
    // so deterministic ordering is stable across all bucket sets.
    if (isSingleton != o.isSingleton)
      return !isSingleton; // false < true
    if (!isSingleton)
      return groupIndex < o.groupIndex;
    return singletonName < o.singletonName;
  }
};

// One layout candidate for a body-op position. `useMux == false` means
// a single fabric.op merging every peer's op name into one op_list
// (must all share one share group). `useMux == true` means one
// fabric.op per share-group bucket fed from the same operands and
// joined by a fresh fabric.mux. The CostModel evaluates the resulting
// sub-FU and the lowest-cost legal candidate wins.
struct OpNodeDecision {
  bool useMux = false;
  // Buckets[k] = the sorted set of op names belonging to share-group k.
  // Each multi-member group (e.g. arith.addi/subi) contributes one
  // entry; each distinct singleton contributes its own entry. For
  // useMux == false the map has exactly one entry.
  ::std::map<BucketKey, ::std::set<::std::string>> buckets;
};

// Group peers by share-group key. Each multi-member group collapses
// its members under one key; each distinct singleton receives its own
// key (so two distinct singletons at the same anchor position never
// land in one bucket).
::std::map<BucketKey, ::std::set<::std::string>>
bucketBySharegroup(const PeerVec &peers,
                   ::llvm::ArrayRef<::fabric::ConfiguredFunction> functions) {
  ::std::map<BucketKey, ::std::set<::std::string>> out;
  for (auto [i, source] : ::llvm::enumerate(peers)) {
    const auto *node = nodeFor(functions[i], source);
    if (!node)
      continue;
    ::llvm::StringRef name = node->operationName;
    out[BucketKey::forName(name)].insert(name.str());
  }
  return out;
}

// Decide the cheapest legal layout for a body-op peer set. Returns
// std::nullopt when no legal layout exists (e.g. cross-share-group
// peers and `allowMux == false`).
::std::optional<OpNodeDecision>
decideOpNode(const PeerVec &peers,
             ::llvm::ArrayRef<::fabric::ConfiguredFunction> functions,
             bool allowMux, const AreaWeights &weights, unsigned bw) {
  auto buckets = bucketBySharegroup(peers, functions);
  if (buckets.empty())
    return std::nullopt;

  // Each singleton has its own BucketKey, so any singleton bucket holds
  // exactly one op name; OpOp::verify's "singletons must occupy
  // fabric.op alone" rule is therefore satisfied by construction. No
  // additional bucket-level filtering is needed here.

  if (buckets.size() == 1) {
    OpNodeDecision d;
    d.useMux = false;
    d.buckets = std::move(buckets);
    return d;
  }
  // More than one bucket: cross-share-group (multi-member groups vs
  // each other, or distinct singletons against any other bucket). Only
  // legal when the user opted into intra-position muxing.
  if (!allowMux)
    return std::nullopt;
  OpNodeDecision d;
  d.useMux = true;
  d.buckets = std::move(buckets);
  // Cost ranking is trivial here -- there is only one cross-share-group
  // layout for tier A: one fabric.op per bucket + one fabric.mux
  // joining them. Future strategies may produce alternative layouts
  // (e.g. per-arm demuxing); the explicit cost call still belongs to
  // the spec contract so we keep the AreaWeights / bw parameters even
  // though the comparison is degenerate.
  (void)weights;
  (void)bw;
  return d;
}

//===----------------------------------------------------------------------===//
// Anchor BFS state.
//===----------------------------------------------------------------------===//

// All fabric.op results for one BodyOp peer set, kept around so that
// multiple anchors naming distinct resultIndex of the same source op
// share one fabric.op emission.
struct EmittedOp {
  ::llvm::SmallVector<::mlir::Value, 2> results;
};

struct PendingOp {
  ::mlir::Operation *placeholder = nullptr;
  ::llvm::SmallVector<::mlir::Value, 2> results;
};

struct AnchorState {
  AnchorState(::mlir::MLIRContext *c, ::mlir::Location l) : ctx(c), loc(l) {}
  ::mlir::MLIRContext *ctx = nullptr;
  ::mlir::Location loc;
  ::llvm::ArrayRef<::fabric::ConfiguredFunction> functions;
  // The wrapper (a func.func) and the inner fabric.fu.
  ::mlir::OpBuilder *bodyBuilder = nullptr;
  ::fabric::FuOp fu;
  // Wrapper-input port -> entry-block argument value of the inner FU.
  ::llvm::DenseMap<unsigned, ::mlir::Value> portValues;
  // Cache: peer set -> emitted Value. Dedups DAG fanout.
  ::llvm::DenseMap<PeerKey, EmittedSlot, PeerKeyInfo> visited;
  // Cache keyed on the resultIndex-stripped peer set, holding the full
  // fabric.op result list. Lets distinct (BodyOp,resultIndex) anchors
  // share one fabric.op emission.
  ::llvm::DenseMap<PeerKey, EmittedOp, PeerKeyInfo> emittedOps;
  // Typed temporary results for an operation currently being emitted. A
  // recursive lookup denotes a legal graph-region feedback edge.
  ::llvm::DenseMap<PeerKey, PendingOp, PeerKeyInfo> pendingOps;
  // Independent physical resource positions and their complete semantic
  // mode choices. Global valid encodings are explicit combinations of these
  // resource-level tuples, never products of individual attribute fields.
  ::llvm::SmallVector<EncodingChoiceGroup, 8> choiceGroups;
  // Diagnostic notes accumulated during the run.
  ::llvm::SmallVector<::std::string, 4> notes;
};

// Forward declarations.
struct EmitOutcome {
  bool ok = false;
  SynthFailureReason reason = SynthFailureReason::None;
  ::mlir::Value value;
};

EmitOutcome materializePeers(AnchorState &st, const PeerVec &peers,
                             const ::loom::SynthConfig &cfg);

// Outcome of emitting one body-op position; returns ALL fabric.op
// result values (lifted to fabric.bits<N>) so callers can project the
// resultIndex of interest.
struct EmitOpOutcome {
  bool ok = false;
  SynthFailureReason reason = SynthFailureReason::None;
  ::llvm::SmallVector<::mlir::Value, 2> results;
  EncodingChoiceGroup choices;
};

// Compute the physical type for every source operation result. Peers must
// agree on result count; each port uses the maximum semantic payload width at
// that position.
::std::optional<::llvm::SmallVector<::mlir::Type, 2>> liftedResultTypesForPeers(
    ::mlir::MLIRContext *ctx, const PeerVec &peers,
    ::llvm::ArrayRef<::fabric::ConfiguredFunction> functions) {
  if (peers.empty())
    return std::nullopt;
  const auto *first = nodeFor(functions.front(), peers.front());
  if (!first)
    return std::nullopt;
  unsigned nr = first->functionType.getNumResults();
  ::llvm::SmallVector<unsigned, 2> bws;
  bws.reserve(nr);
  for (unsigned i = 0; i < nr; ++i) {
    auto bw = tryBitWidthOf(first->functionType.getResult(i));
    if (!bw.has_value())
      return std::nullopt;
    bws.push_back(*bw);
  }
  for (auto [peerIndex, source] : ::llvm::enumerate(peers)) {
    const auto *node = nodeFor(functions[peerIndex], source);
    if (!node || node->functionType.getNumResults() != nr)
      return std::nullopt;
    for (unsigned i = 0; i < nr; ++i) {
      auto bw = tryBitWidthOf(node->functionType.getResult(i));
      if (!bw.has_value())
        return std::nullopt;
      bws[i] = std::max(bws[i], *bw);
    }
  }
  ::llvm::SmallVector<::mlir::Type, 2> out;
  out.reserve(nr);
  for (unsigned w : bws)
    out.push_back(::fabric::BitsType::get(ctx, w));
  return out;
}

// Emit a fabric.op (or per-bucket fabric.op + fabric.mux) for a body-op
// peer set whose operands have already been materialized into
// `operandValues`. Returns the FULL list of result values from the
// emitted fabric.op (or a single mux output for the cross-share-group
// case, which is single-result by construction).
EmitOpOutcome
emitBodyOpPositionMulti(AnchorState &st, const OpNodeDecision &decision,
                        const PeerVec &peers, ::mlir::ValueRange operandValues,
                        ::llvm::ArrayRef<::mlir::Type> resultTypes) {
  EmitOpOutcome r;
  if (!decision.useMux) {
    const auto &kv = *decision.buckets.begin();
    ::mlir::ArrayAttr opList = sortedOpListFor(kv.second, st.ctx);
    auto op = emitFabricOp(*st.bodyBuilder, st.loc, opList, operandValues,
                           resultTypes);
    for (auto [peerIndex, source] : ::llvm::enumerate(peers)) {
      const auto *node = nodeFor(st.functions[peerIndex], source);
      if (!node) {
        r.reason = SynthFailureReason::TopologyMismatch;
        return r;
      }
      auto assignment = opModeAssignment(op, *node, st.ctx);
      if (!assignment) {
        r.reason = SynthFailureReason::TopologyMismatch;
        return r;
      }
      EncodingFragment fragment;
      fragment.push_back(*assignment);
      addUniqueChoice(r.choices, std::move(fragment));
    }
    r.ok = true;
    r.results.assign(op.getOutputs().begin(), op.getOutputs().end());
    return r;
  }

  if (resultTypes.size() != 1) {
    r.reason = SynthFailureReason::TopologyMismatch;
    return r;
  }

  unsigned branchCount = decision.buckets.size();
  ::llvm::SmallVector<::fabric::DemuxOp, 4> inputDemuxes;
  for (::mlir::Value operand : operandValues)
    inputDemuxes.push_back(
        emitFabricDemux(*st.bodyBuilder, st.loc, operand, branchCount));

  ::std::map<BucketKey, unsigned> branchByBucket;
  ::std::map<BucketKey, ::fabric::OpOp> opByBucket;
  ::llvm::SmallVector<::mlir::Value, 4> arms;
  arms.reserve(branchCount);
  unsigned branch = 0;
  for (const auto &kv : decision.buckets) {
    branchByBucket[kv.first] = branch;
    ::mlir::ArrayAttr opList = sortedOpListFor(kv.second, st.ctx);
    ::llvm::SmallVector<::mlir::Value, 4> branchOperands;
    for (::fabric::DemuxOp demux : inputDemuxes)
      branchOperands.push_back(demux.getOutputs()[branch]);
    auto opOp = emitFabricOp(*st.bodyBuilder, st.loc, opList, branchOperands,
                             resultTypes);
    opByBucket[kv.first] = opOp;
    arms.push_back(opOp.getOutputs()[0]);
    ++branch;
  }
  auto mux = emitFabricMux(*st.bodyBuilder, st.loc, arms, resultTypes[0]);

  for (auto [peerIndex, source] : ::llvm::enumerate(peers)) {
    const auto *node = nodeFor(st.functions[peerIndex], source);
    if (!node) {
      r.reason = SynthFailureReason::TopologyMismatch;
      return r;
    }
    BucketKey bucket = BucketKey::forName(node->operationName);
    unsigned selected = branchByBucket.at(bucket);
    EncodingFragment fragment;
    for (::fabric::DemuxOp demux : inputDemuxes)
      fragment.push_back(routeAssignment(demux, selected, st.ctx));
    auto assignment = opModeAssignment(opByBucket.at(bucket), *node, st.ctx);
    if (!assignment) {
      r.reason = SynthFailureReason::TopologyMismatch;
      return r;
    }
    fragment.push_back(*assignment);
    fragment.push_back(routeAssignment(mux, selected, st.ctx));
    addUniqueChoice(r.choices, std::move(fragment));
  }

  r.ok = true;
  r.results.push_back(mux.getOutput());
  return r;
}

EmitOutcome materializePeers(AnchorState &st, const PeerVec &peers,
                             const ::loom::SynthConfig &cfg) {
  EmitOutcome out;

  // Dedup: the same peer set may be reached from multiple parents
  // (DAG fanout). Reuse the previously emitted Value verbatim.
  PeerKey key = keyOf(peers);
  auto it = st.visited.find(key);
  if (it != st.visited.end()) {
    out.ok = true;
    out.value = it->second.value;
    return out;
  }

  if (!peersUniformKind(peers)) {
    out.reason = SynthFailureReason::TopologyMismatch;
    return out;
  }

  auto kind = peers.front().kind;
  if (kind == ::fabric::ConfiguredValue::Kind::InputPort) {
    unsigned port = peers.front().index;
    for (const ::fabric::ConfiguredValue &value : peers)
      if (value.index != port) {
        out.reason = SynthFailureReason::TopologyMismatch;
        return out;
      }
    ::mlir::Value physicalInput = st.portValues.lookup(port);
    if (!physicalInput) {
      out.reason = SynthFailureReason::TopologyMismatch;
      return out;
    }
    out.ok = true;
    out.value = physicalInput;
    st.visited[key] = EmittedSlot{out.value};
    return out;
  }

  // Validate share-group and arity, then size the physical payload to
  // the maximum semantic width at this graph position.
  unsigned bw = 0;
  if (!collectPeerWidth(peers, st.functions, bw)) {
    out.reason = SynthFailureReason::TopologyMismatch;
    return out;
  }
  unsigned arity = 0;
  if (!peersUniformArity(peers, st.functions, arity)) {
    out.reason = SynthFailureReason::TopologyMismatch;
    return out;
  }
  if (!peersShareStreamStepKind(peers, st.functions)) {
    st.notes.push_back(
        "anchor: dataflow.stream peers require one fixed step_kind");
    out.reason = SynthFailureReason::TopologyMismatch;
    return out;
  }

  // The peer set may have been emitted before under a different
  // resultIndex (e.g. yield #0 saw `dataflow.stream#0`, yield #1 saw
  // `dataflow.stream#1`). Reuse the same fabric.op by projecting the
  // requested resultIndex from the cached result list.
  PeerKey opKey = opKeyOf(peers);
  auto reuse = st.emittedOps.find(opKey);
  if (reuse != st.emittedOps.end()) {
    unsigned ri = peers.front().result;
    if (ri >= reuse->second.results.size()) {
      out.reason = SynthFailureReason::TopologyMismatch;
      return out;
    }
    out.ok = true;
    out.value = reuse->second.results[ri];
    st.visited[key] = EmittedSlot{out.value};
    return out;
  }
  auto pending = st.pendingOps.find(opKey);
  if (pending != st.pendingOps.end()) {
    unsigned ri = peers.front().result;
    if (ri >= pending->second.results.size()) {
      out.reason = SynthFailureReason::TopologyMismatch;
      return out;
    }
    out.ok = true;
    out.value = pending->second.results[ri];
    return out;
  }

  // Pre-decision: distinct share-groups when intra-position muxing is
  // disabled is a `cross_share_group` failure (distinct from
  // `topology_mismatch`).
  auto buckets = bucketBySharegroup(peers, st.functions);
  if (buckets.size() > 1 && !cfg.anchorAllowIntraPositionMux) {
    out.reason = SynthFailureReason::CrossShareGroup;
    return out;
  }

  auto resultTypesOpt = liftedResultTypesForPeers(st.ctx, peers, st.functions);
  if (!resultTypesOpt.has_value()) {
    out.reason = SynthFailureReason::TopologyMismatch;
    return out;
  }
  ::mlir::OperationState placeholderState(
      st.loc, ::mlir::UnrealizedConversionCastOp::getOperationName());
  placeholderState.addTypes(*resultTypesOpt);
  ::mlir::Operation *placeholder = st.bodyBuilder->create(placeholderState);
  PendingOp pendingSlot;
  pendingSlot.placeholder = placeholder;
  pendingSlot.results.assign(placeholder->getResults().begin(),
                             placeholder->getResults().end());
  st.pendingOps[opKey] = std::move(pendingSlot);

  // Recurse through exact ordered operands.
  ::llvm::SmallVector<::mlir::Value, 4> operandValues;
  operandValues.reserve(arity);
  for (unsigned i = 0; i < arity; ++i) {
    PeerVec child;
    child.reserve(peers.size());
    for (auto [peerIndex, source] : ::llvm::enumerate(peers)) {
      const auto *node = nodeFor(st.functions[peerIndex], source);
      if (!node) {
        out.reason = SynthFailureReason::TopologyMismatch;
        return out;
      }
      child.push_back(node->operands[i]);
    }
    EmitOutcome co = materializePeers(st, child, cfg);
    if (!co.ok) {
      out.reason = co.reason;
      return out;
    }
    operandValues.push_back(co.value);
  }

  AreaWeights weights;
  weights.muxPenalty = cfg.costMuxPenalty;
  weights.demuxPenalty = cfg.costDemuxPenalty;
  weights.carryPenalty = cfg.costCarryPenalty;
  auto decision = decideOpNode(peers, st.functions,
                               cfg.anchorAllowIntraPositionMux, weights, bw);
  if (!decision.has_value()) {
    if (buckets.size() > 1)
      out.reason = SynthFailureReason::CrossShareGroup;
    else
      out.reason = SynthFailureReason::TopologyMismatch;
    return out;
  }
  EmitOpOutcome emitted = emitBodyOpPositionMulti(
      st, *decision, peers, operandValues, *resultTypesOpt);
  if (!emitted.ok) {
    out.reason = emitted.reason != SynthFailureReason::None
                     ? emitted.reason
                     : SynthFailureReason::TopologyMismatch;
    return out;
  }
  if (emitted.choices.choices.empty()) {
    out.reason = SynthFailureReason::TopologyMismatch;
    return out;
  }
  st.choiceGroups.push_back(std::move(emitted.choices));
  unsigned ri = peers.front().result;
  if (ri >= emitted.results.size()) {
    out.reason = SynthFailureReason::TopologyMismatch;
    return out;
  }
  EmittedOp slot;
  slot.results.assign(emitted.results.begin(), emitted.results.end());
  PendingOp &pendingResult = st.pendingOps.at(opKey);
  if (pendingResult.results.size() != slot.results.size()) {
    out.reason = SynthFailureReason::TopologyMismatch;
    return out;
  }
  for (auto [temporary, actual] :
       ::llvm::zip(pendingResult.results, slot.results))
    temporary.replaceAllUsesWith(actual);
  pendingResult.placeholder->erase();
  st.pendingOps.erase(opKey);
  st.emittedOps[opKey] = slot;
  out.ok = true;
  out.value = emitted.results[ri];
  st.visited[key] = EmittedSlot{out.value};
  return out;
}

//===----------------------------------------------------------------------===//
// Wrapper builder.
//===----------------------------------------------------------------------===//

// Wrapper symbol name follows the spec: `fu_<sanitized(group)>` with
// sanitization replacing non-`[A-Za-z0-9_]` characters by `_`.
::std::string wrapperName(::llvm::StringRef groupName) {
  ::std::string out = "fu_";
  for (char c : groupName) {
    bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
              (c >= '0' && c <= '9') || c == '_';
    out.push_back(ok ? c : '_');
  }
  return out;
}

::std::optional<::mlir::ArrayAttr> buildValidEncodings(AnchorState &state,
                                                       unsigned outputCount,
                                                       ::std::string &error) {
  ::llvm::DenseMap<::mlir::Operation *, unsigned> resourceIndex;
  for (auto [index, op] :
       ::llvm::enumerate(state.fu.getBody().front().without_terminator()))
    resourceIndex[&op] = index;

  auto fragmentKey = [&](const EncodingFragment &fragment) {
    ::std::string key;
    ::llvm::raw_string_ostream os(key);
    ::llvm::SmallVector<const PendingAssignment *, 8> ordered;
    for (const PendingAssignment &assignment : fragment)
      ordered.push_back(&assignment);
    ::llvm::sort(ordered, [&](const PendingAssignment *lhs,
                              const PendingAssignment *rhs) {
      return resourceIndex.lookup(lhs->resource) <
             resourceIndex.lookup(rhs->resource);
    });
    for (const PendingAssignment *assignment : ordered)
      os << resourceIndex.lookup(assignment->resource) << ':'
         << printAttribute(assignment->payload) << ';';
    return os.str();
  };

  ::llvm::SmallVector<EncodingFragment, 8> combinations(1);
  for (EncodingChoiceGroup &group : state.choiceGroups) {
    ::llvm::sort(group.choices,
                 [&](const EncodingFragment &lhs, const EncodingFragment &rhs) {
                   return fragmentKey(lhs) < fragmentKey(rhs);
                 });
    ::llvm::SmallVector<EncodingFragment, 8> next;
    for (const EncodingFragment &base : combinations) {
      for (const EncodingFragment &choice : group.choices) {
        EncodingFragment merged(base.begin(), base.end());
        for (const PendingAssignment &assignment : choice) {
          if (::llvm::any_of(merged, [&](const PendingAssignment &existing) {
                return existing.resource == assignment.resource;
              })) {
            error = "independent encoding choices share a physical resource";
            return std::nullopt;
          }
          merged.push_back(assignment);
        }
        next.push_back(std::move(merged));
      }
    }
    combinations = std::move(next);
  }

  if (combinations.empty()) {
    error = "synthesis produced no valid semantic encoding";
    return std::nullopt;
  }

  auto *context = state.ctx;
  auto i32 = ::mlir::IntegerType::get(context, 32);
  ::llvm::SmallVector<::mlir::Attribute, 4> outputs;
  for (unsigned output = 0; output < outputCount; ++output)
    outputs.push_back(::mlir::IntegerAttr::get(i32, output));
  auto outputArray = ::mlir::ArrayAttr::get(context, outputs);

  ::llvm::SmallVector<::mlir::Attribute, 8> encodings;
  std::string firstProjectionError;
  for (EncodingFragment &combination : combinations) {
    ::llvm::sort(combination, [&](const PendingAssignment &lhs,
                                  const PendingAssignment &rhs) {
      return resourceIndex.lookup(lhs.resource) <
             resourceIndex.lookup(rhs.resource);
    });
    ::llvm::SmallVector<::mlir::Attribute, 12> resources;
    for (const PendingAssignment &assignment : combination) {
      ::llvm::SmallVector<::mlir::NamedAttribute, 8> fields;
      fields.emplace_back(::mlir::StringAttr::get(context, "resource"),
                          ::mlir::IntegerAttr::get(
                              i32, resourceIndex.lookup(assignment.resource)));
      fields.append(assignment.payload.begin(), assignment.payload.end());
      resources.push_back(::mlir::DictionaryAttr::get(context, fields));
    }
    ::llvm::SmallVector<::mlir::NamedAttribute, 2> fields = {
        {::mlir::StringAttr::get(context, "outputs"), outputArray},
        {::mlir::StringAttr::get(context, "resources"),
         ::mlir::ArrayAttr::get(context, resources)}};
    auto encoding = ::mlir::DictionaryAttr::get(context, fields);
    ::fabric::ConfiguredFunction projected;
    std::string projectionError;
    if (::mlir::failed(::fabric::projectConfiguredFunction(
            state.fu, encoding, projected, projectionError))) {
      if (firstProjectionError.empty())
        firstProjectionError = std::move(projectionError);
      continue;
    }
    encodings.push_back(encoding);
  }
  if (encodings.empty()) {
    error = "synthesis produced no type-coherent valid semantic encoding";
    if (!firstProjectionError.empty())
      error += ": " + firstProjectionError;
    return std::nullopt;
  }
  return ::mlir::ArrayAttr::get(context, encodings);
}

} // namespace

//===----------------------------------------------------------------------===//
// Anchor synthesis producer.
//===----------------------------------------------------------------------===//

static SynthResult runAnchorSynthesis(const ::loom::SynthConfig &cfg,
                                      const SynthInputs &inputs) {
  SynthResult result;

  if (inputs.functions.empty()) {
    result.failureReason = SynthFailureReason::InvalidInput;
    result.notes.push_back("anchor: no input functions in synth group");
    return result;
  }
  if (!inputs.context) {
    result.failureReason = SynthFailureReason::InvalidInput;
    result.notes.push_back("anchor: missing scratch MLIRContext");
    return result;
  }

  ::mlir::MLIRContext *ctx = inputs.context;
  ::mlir::Location loc = ::mlir::UnknownLoc::get(ctx);

  ::llvm::SmallVector<::llvm::SmallVector<::fabric::ConfiguredValue, 4>, 4>
      perInputAnchors;
  perInputAnchors.reserve(inputs.functions.size());
  unsigned arity = 0;
  for (auto [i, function] : ::llvm::enumerate(inputs.functions)) {
    ::llvm::SmallVector<::fabric::ConfiguredValue, 4> anchors;
    for (const ::fabric::ConfiguredBoundaryOutput &output : function.outputs)
      anchors.push_back(output.value);
    if (i == 0)
      arity = static_cast<unsigned>(anchors.size());
    else if (anchors.size() != arity) {
      result.failureReason = SynthFailureReason::TopologyMismatch;
      result.notes.push_back(
          "anchor: input functions disagree on result arity");
      return result;
    }
    perInputAnchors.push_back(std::move(anchors));
  }
  if (arity == 0) {
    result.failureReason = SynthFailureReason::TopologyMismatch;
    result.notes.push_back("anchor: empty result list");
    return result;
  }

  // Wrapper input ports: one per block-argument index, sized to the maximum
  // software payload width at that position.
  auto portsOpt = collectWrapperInputSlots(inputs.functions);
  if (!portsOpt.has_value()) {
    result.failureReason = SynthFailureReason::TopologyMismatch;
    result.notes.push_back(
        "anchor: input functions disagree on boundary shape or type kind");
    return result;
  }
  auto &ports = *portsOpt;

  // Build the wrapper fabric.module + inner fabric.pe + fabric.fu skeleton.
  // Two type lanes:
  //   * "inner" types: the actual lifted bit-widths per input slot and per
  //     yield slot. Used for FU body block-arg types (input lane) and FU
  //     body yield value types (output lane).
  //   * "outer" types: the uniform PE port width W = max(all lifted widths)
  //     applied to every PE input port, every PE result port, and every
  //     FU outer input/result type. Bridging to inner widths happens at
  //     the FU boundary via the input-side `to <inner-type>` truncation
  //     and the output-side `to <outer-type>` widening on fabric.yield.

  // Per-slot lifted (inner) input types.
  ::llvm::SmallVector<::mlir::Type, 4> innerInputTypes;
  innerInputTypes.reserve(ports.size());
  for (const WrapperInputSlot &p : ports)
    innerInputTypes.push_back(::fabric::BitsType::get(ctx, p.bitwidth));

  // Per-slot physical output types use the maximum software payload width at
  // each result position.
  ::llvm::SmallVector<::mlir::Type, 4> innerResultTypes;
  innerResultTypes.reserve(arity);
  auto outputWidths = collectWrapperOutputWidths(inputs.functions);
  if (!outputWidths) {
    result.failureReason = SynthFailureReason::TopologyMismatch;
    result.notes.push_back("anchor: yield operand has unsupported type");
    return result;
  }
  for (unsigned width : *outputWidths)
    innerResultTypes.push_back(::fabric::BitsType::get(ctx, width));

  // Pick uniform W = max over all lifted input AND output widths. The
  // PE-uniform-width invariant constrains the PE port-list types and
  // the FU outer port types to a single bits<W>. When all widths are
  // 0, W = 0 (legitimate for none-only signatures).
  unsigned uniformW = 0;
  for (const WrapperInputSlot &p : ports)
    uniformW = std::max(uniformW, p.bitwidth);
  for (::mlir::Type t : innerResultTypes) {
    if (auto bt = ::llvm::dyn_cast<::fabric::BitsType>(t))
      uniformW = std::max(uniformW, bt.getWidth());
  }
  ::mlir::Type uniformBits = ::fabric::BitsType::get(ctx, uniformW);

  // PE/FU outer (port-uniform) types.
  ::llvm::SmallVector<::mlir::Type, 4> wrapperInputTypes(ports.size(),
                                                         uniformBits);
  ::llvm::SmallVector<::mlir::Type, 4> wrapperResultTypes(arity, uniformBits);

  ::std::string symName = wrapperName(inputs.groupName);

  // 1. Build the outer fabric.module. The module declares the inputs as
  // entry-block arguments and carries no SSA results (the inner
  // fabric.pe owns the visible results). The module's body terminator
  // is a zero-operand fabric.yield.
  ::llvm::SmallVector<::mlir::Type, 0> moduleResultTypes;
  auto moduleFuncType = ::mlir::FunctionType::get(
      ctx, wrapperInputTypes, ::mlir::TypeRange(moduleResultTypes));
  ::mlir::OperationState moduleState(loc,
                                     ::fabric::ModuleOp::getOperationName());
  moduleState.addAttribute("sym_name", ::mlir::StringAttr::get(ctx, symName));
  moduleState.addAttribute("function_type",
                           ::mlir::TypeAttr::get(moduleFuncType));
  ::mlir::Region *moduleRegion = moduleState.addRegion();
  ::mlir::Block *moduleEntry = new ::mlir::Block();
  moduleRegion->push_back(moduleEntry);
  ::llvm::SmallVector<::mlir::Location, 4> moduleArgLocs(
      wrapperInputTypes.size(), loc);
  moduleEntry->addArguments(wrapperInputTypes, moduleArgLocs);
  ::mlir::OpBuilder topBuilder(ctx);
  ::mlir::Operation *rawModule = topBuilder.create(moduleState);
  auto wrapper = ::mlir::cast<::fabric::ModuleOp>(rawModule);

  // 2. Build the inner fabric.pe. Spatial schedule, anonymous form: PE
  // operands are the module's entry-block arguments, PE results match
  // the FU's results (or, when the FU has no results, a single bits<W>
  // placeholder so PE's L>=1 invariant is satisfied).
  ::mlir::OpBuilder moduleBuilder(moduleEntry, moduleEntry->end());
  ::llvm::SmallVector<::mlir::Type, 4> peResultTypes(wrapperResultTypes);
  if (peResultTypes.empty()) {
    // FU has zero results. PE still needs L>=1; declare one bits<W>
    // output port that the FU does not drive. Width derived from the
    // module's input width (uniform W) when present, else 0 (which is
    // a verifier error for empty-input wrappers, but anchor already
    // requires K>=1 above).
    unsigned w =
        wrapperInputTypes.empty()
            ? 0u
            : ::llvm::cast<::fabric::BitsType>(wrapperInputTypes[0]).getWidth();
    peResultTypes.push_back(::fabric::BitsType::get(ctx, w));
  }
  ::mlir::OperationState peState(loc, ::fabric::PeOp::getOperationName());
  peState.addOperands(::mlir::ValueRange(moduleEntry->getArguments()));
  peState.addTypes(peResultTypes);
  peState.addAttribute("schedule", ::fabric::ScheduleAttr::get(
                                       ctx, ::fabric::Schedule::Spatial));
  ::mlir::Region *peRegion = peState.addRegion();
  ::mlir::Block *peEntry = new ::mlir::Block();
  peRegion->push_back(peEntry);
  ::llvm::SmallVector<::mlir::Location, 4> peArgLocs(wrapperInputTypes.size(),
                                                     loc);
  peEntry->addArguments(wrapperInputTypes, peArgLocs);
  ::mlir::Operation *rawPe = moduleBuilder.create(peState);
  auto pe = ::mlir::cast<::fabric::PeOp>(rawPe);
  (void)pe;

  // 3. Build the inner fabric.fu inside the PE body. FU operands are
  // the PE's block arguments (outer bits<W>); FU outer result types are
  // also bits<W>. FU body block-arg types are the per-slot inner widths
  // (input-side truncation when inner < outer is handled by the FU
  // boundary `to <inner-type>` clause).
  ::mlir::OpBuilder peBodyBuilder(peEntry, peEntry->end());
  ::mlir::OperationState fuState(loc, ::fabric::FuOp::getOperationName());
  fuState.addOperands(::mlir::ValueRange(peEntry->getArguments()));
  fuState.addTypes(wrapperResultTypes);
  ::mlir::Region *fuRegion = fuState.addRegion();
  ::mlir::Block *fuEntry = new ::mlir::Block();
  fuRegion->push_back(fuEntry);
  ::llvm::SmallVector<::mlir::Location, 4> fuArgLocs(innerInputTypes.size(),
                                                     loc);
  fuEntry->addArguments(innerInputTypes, fuArgLocs);
  ::mlir::Operation *rawFu = peBodyBuilder.create(fuState);
  auto fu = ::mlir::cast<::fabric::FuOp>(rawFu);

  // 4. Build the FU body using the BFS algorithm. Anchor strategy is
  // self-contained: every node it emits inside the body lives at the
  // top level (no nested control flow), so a single OpBuilder anchored
  // at the FU entry block suffices.
  ::mlir::OpBuilder bodyBuilder(fuEntry, fuEntry->end());

  AnchorState st(ctx, loc);
  st.functions = inputs.functions;
  st.bodyBuilder = &bodyBuilder;
  st.fu = fu;
  st.portValues.reserve(fuEntry->getNumArguments());
  for (auto [slot, argument] : ::llvm::zip(ports, fuEntry->getArguments()))
    st.portValues[slot.port] = argument;

  // 5. Walk yield anchors in order, materializing each peer set. The
  // BFS dedups DAG fanout via `visited`, so emitting in anchor order
  // does not duplicate inner ops shared across multiple yield outputs.
  ::llvm::SmallVector<::mlir::Value, 4> yieldValues;
  yieldValues.reserve(arity);
  for (unsigned k = 0; k < arity; ++k) {
    PeerVec peers;
    peers.reserve(inputs.functions.size());
    for (auto &perInput : perInputAnchors)
      peers.push_back(perInput[k]);
    EmitOutcome out = materializePeers(st, peers, cfg);
    if (!out.ok) {
      // Drop the partially built wrapper deterministically. The
      // worker-local context goes out of scope in the caller.
      wrapper.erase();
      result.failureReason = out.reason != SynthFailureReason::None
                                 ? out.reason
                                 : SynthFailureReason::TopologyMismatch;
      for (auto &n : st.notes)
        result.notes.push_back(std::move(n));
      return result;
    }
    yieldValues.push_back(out.value);
  }

  // 6. Emit the FU's terminator fabric.yield (matches the FU result
  // arity, which may be zero) and the module-level fabric.yield (zero
  // operands; fabric.module declares no SSA results).
  //
  // For each yield value `k`, when the inner bit-width is narrower than
  // the FU outer width W, attach the per-value `to <outer>` clause via
  // the `declared_types` array attribute. This drives the FU output
  // boundary widening (low-bit-aligned, high bits zero-filled).
  ::mlir::OperationState fuYieldState(loc,
                                      ::fabric::YieldOp::getOperationName());
  fuYieldState.addOperands(yieldValues);
  if (!yieldValues.empty()) {
    ::llvm::SmallVector<::mlir::Attribute, 4> declaredAttrs;
    declaredAttrs.reserve(yieldValues.size());
    for (auto [k, v] : ::llvm::enumerate(yieldValues)) {
      // The declared destination is always the FU's outer result type
      // (uniform bits<W>). When inner == outer the verifier still
      // accepts the redundant annotation, but to keep the printed IR
      // clean we record the declared type only when it actually
      // differs from the inner SSA type.
      ::mlir::Type declared = wrapperResultTypes[k];
      declaredAttrs.push_back(::mlir::TypeAttr::get(declared));
    }
    bool anyWidens = false;
    for (auto [k, v] : ::llvm::enumerate(yieldValues)) {
      if (v.getType() != wrapperResultTypes[k]) {
        anyWidens = true;
        break;
      }
    }
    if (anyWidens)
      fuYieldState.addAttribute("declared_types",
                                ::mlir::ArrayAttr::get(ctx, declaredAttrs));
  }
  bodyBuilder.create(fuYieldState);

  ::std::string encodingError;
  auto encodings = buildValidEncodings(st, arity, encodingError);
  if (!encodings) {
    wrapper.erase();
    result.failureReason = SynthFailureReason::VerifierFailed;
    result.notes.push_back("anchor: " + encodingError);
    return result;
  }
  fu->setAttr("valid_encodings", *encodings);

  ::mlir::OperationState moduleYieldState(
      loc, ::fabric::YieldOp::getOperationName());
  moduleBuilder.create(moduleYieldState);

  // Wrap into an OwningOpRef so the worker-side thread retains
  // ownership until the main thread re-homes it.
  ::mlir::OwningOpRef<::fabric::ModuleOp> owned(wrapper);

  result.wrapper = std::move(owned);
  for (auto &n : st.notes)
    result.notes.push_back(std::move(n));
  return result;
}

static ::fabric::FuOp findInnerFu(::fabric::ModuleOp wrapper) {
  if (!wrapper)
    return nullptr;
  ::fabric::FuOp found;
  wrapper.walk([&](::fabric::FuOp fu) {
    if (!found)
      found = fu;
  });
  return found;
}

static void enforceCanonicalAcceptance(
    SynthResult &result, ::llvm::ArrayRef<::fabric::ConfiguredFunction> inputs,
    const ::loom::SynthConfig &cfg) {
  if (!result.success())
    return;

  ::fabric::FuOp fu = findInnerFu(result.wrapper.get());
  if (!fu || ::mlir::failed(::mlir::verify(result.wrapper.get()))) {
    result.wrapper = nullptr;
    result.failureReason = SynthFailureReason::VerifierFailed;
    result.notes.push_back(
        "canonical synthesis gate: wrapper or FU verification failed");
    return;
  }

  CoverageVerifier verifier(cfg);
  result.coverage = verifier.verify(fu, inputs);
  if (!result.coverage.allCovered()) {
    result.wrapper = nullptr;
    result.failureReason = SynthFailureReason::VerifierFailed;
    result.notes.push_back(
        "canonical synthesis gate: explicit encodings do not cover every "
        "input function");
    return;
  }
  result.capability = measureCapability(fu, result.coverage);
}

SynthResult synthesize(const ::loom::SynthConfig &cfg,
                       const SynthInputs &inputs) {
  if (cfg.strategy != "anchor") {
    SynthResult result;
    result.failureReason = SynthFailureReason::InvalidInput;
    std::string note;
    ::llvm::raw_string_ostream os(note);
    os << "unknown strategy '" << cfg.strategy << "'";
    os.flush();
    result.notes.push_back(std::move(note));
    return result;
  }

  SynthResult result = runAnchorSynthesis(cfg, inputs);
  enforceCanonicalAcceptance(result, inputs.functions, cfg);
  return result;
}

//===----------------------------------------------------------------------===//
// Public wrapper helpers.
//===----------------------------------------------------------------------===//

::std::optional<WrapperPorts>
collectWrapperPorts(::llvm::ArrayRef<::fabric::ConfiguredFunction> functions,
                    ::mlir::MLIRContext *ctx) {
  if (!ctx)
    return ::std::nullopt;
  auto inputSlots = collectWrapperInputSlots(functions);
  if (!inputSlots.has_value())
    return ::std::nullopt;
  auto outputWidths = collectWrapperOutputWidths(functions);
  if (!outputWidths.has_value())
    return ::std::nullopt;
  unsigned uniformWidth = 0;
  for (const WrapperInputSlot &input : *inputSlots)
    uniformWidth = std::max(uniformWidth, input.bitwidth);
  for (unsigned output : *outputWidths)
    uniformWidth = std::max(uniformWidth, output);
  WrapperPorts ports;
  ports.inputs.reserve(inputSlots->size());
  for (const WrapperInputSlot &input : *inputSlots) {
    (void)input;
    ports.inputs.push_back(::fabric::BitsType::get(ctx, uniformWidth));
  }
  ports.outputs.reserve(outputWidths->size());
  for (unsigned output : *outputWidths) {
    (void)output;
    ports.outputs.push_back(::fabric::BitsType::get(ctx, uniformWidth));
  }
  return ports;
}

} // namespace loom::fabric::tech
