#include "Fabric/IR/MemoryCapabilityDomains.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowServiceSchema.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>

using namespace dataflow;
using namespace dataflow::semantics;
using namespace fabric;

namespace {

constexpr llvm::StringLiteral accessFixture = R"mlir(
module {
  func.func @element_f32(%mem: memref<8xf32>, %address: index, %ctrl: none)
      -> (f32, none) {
    %data, %done = dataflow.load %mem[%address] %ctrl : memref<8xf32>
    return %data, %done : f32, none
  }

  func.func @vector4_f32(%mem: memref<8xf32>, %address: index, %ctrl: none)
      -> (vector<4xf32>, none) {
    %data, %done = dataflow.load %mem[%address] %ctrl
        : memref<8xf32>, vector<4xf32>
    return %data, %done : vector<4xf32>, none
  }

  func.func @vector2_f64(%mem: memref<8xf64>, %address: index, %ctrl: none)
      -> (vector<2xf64>, none) {
    %data, %done = dataflow.load %mem[%address] %ctrl
        : memref<8xf64>, vector<2xf64>
    return %data, %done : vector<2xf64>, none
  }

  func.func @atomic_element_f32(%mem: memref<8xf32>, %address: index,
                                %ctrl: none) -> (f32, none) {
    %data, %done = dataflow.load %mem[%address] %ctrl
        {contract = #dataflow.atomic_access<ordering = acquire,
                                            sync_scope = <system>,
                                            source_alignment_bytes = 4>}
        : memref<8xf32>
    return %data, %done : f32, none
  }
}
)mlir";

[[noreturn]] void fail(llvm::StringRef test, const llvm::Twine &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectRejected(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted a noncanonical declaration");
  llvm::consumeError(value.takeError());
}

UnsignedDomain singleton(std::uint64_t value) {
  return take("singleton",
              UnsignedDomain::fromCanonical({UnsignedInterval{value, value}}));
}

AlignmentDomain scalarAlignment() {
  return take("alignment", AlignmentDomain::create(singleton(0)));
}

AlignmentDomain fourByteAlignment() {
  return take("alignment", AlignmentDomain::create(singleton(2)));
}

ClosedEnumDomain<ReadSubwordSemantics> readExact() {
  return take("read semantics",
              ClosedEnumDomain<ReadSubwordSemantics>::fromCanonical(
                  {ReadSubwordSemantics::Exact}));
}

ClosedEnumDomain<WriteSubwordSemantics> writeNotApplicable() {
  return take("write semantics",
              ClosedEnumDomain<WriteSubwordSemantics>::fromCanonical(
                  {WriteSubwordSemantics::NotApplicable}));
}

MemoryAccessClass accessClass(MemoryAccessForm form, std::uint64_t elementBits,
                              std::uint64_t laneCount) {
  return take("access class",
              MemoryAccessClass::create(
                  form, singleton(elementBits), singleton(laneCount),
                  {MaskInactivePair{MemoryMaskForm::Absent,
                                    InactiveLaneSemantics::NotApplicable}},
                  scalarAlignment(), readExact(), writeNotApplicable()));
}

MemoryAccessClass accessClass(MemoryAccessForm form, UnsignedDomain elementBits,
                              UnsignedDomain laneCounts) {
  return take("access class",
              MemoryAccessClass::create(
                  form, std::move(elementBits), std::move(laneCounts),
                  {MaskInactivePair{MemoryMaskForm::Absent,
                                    InactiveLaneSemantics::NotApplicable}},
                  scalarAlignment(), readExact(), writeNotApplicable()));
}

void checkUnsignedDomains() {
  UnsignedDomain normalized = take(
      "normalization", UnsignedDomain::normalize(
                           {UnsignedInterval{7, 7}, UnsignedInterval{1, 2},
                            UnsignedInterval{2, 4}, UnsignedInterval{5, 6}}));
  require("normalization", normalized.intervals().size() == 1,
          "overlap and adjacency were not merged");
  require("normalization",
          normalized.intervals().front() == UnsignedInterval{1, 7},
          "the merged interval is not exact");

  UnsignedDomain finite =
      take("finite normalization",
           UnsignedDomain::normalize(
               {UnsignedInterval{9, 9}, UnsignedInterval{2, 2}}));
  require("finite normalization",
          finite.intervals() ==
              llvm::ArrayRef<UnsignedInterval>(
                  {UnsignedInterval{2, 2}, UnsignedInterval{9, 9}}),
          "singleton intervals did not form one sorted finite domain");
  require("membership",
          normalized.contains(1) && normalized.contains(7) &&
              !normalized.contains(8),
          "inclusive membership is wrong");

  expectRejected<UnsignedDomain>("empty", UnsignedDomain::fromCanonical({}));
  expectRejected<UnsignedDomain>(
      "unsorted", UnsignedDomain::fromCanonical(
                      {UnsignedInterval{4, 4}, UnsignedInterval{1, 1}}));
  expectRejected<UnsignedDomain>(
      "overlap", UnsignedDomain::fromCanonical(
                     {UnsignedInterval{1, 3}, UnsignedInterval{3, 5}}));
  expectRejected<UnsignedDomain>(
      "adjacent", UnsignedDomain::fromCanonical(
                      {UnsignedInterval{1, 2}, UnsignedInterval{3, 5}}));
  expectRejected<UnsignedDomain>(
      "reversed", UnsignedDomain::fromCanonical({UnsignedInterval{5, 4}}));

  expectRejected<AlignmentDomain>("alignment range",
                                  AlignmentDomain::create(singleton(64)));
  require("alignment membership",
          scalarAlignment().containsBytes(1) &&
              !scalarAlignment().containsBytes(0) &&
              !scalarAlignment().containsBytes(3),
          "byte alignment membership is not exact");

  expectRejected<MemoryAccessClass>(
      "zero element width",
      MemoryAccessClass::create(
          MemoryAccessForm::Element, singleton(0), singleton(1),
          {MaskInactivePair{MemoryMaskForm::Absent,
                            InactiveLaneSemantics::NotApplicable}},
          scalarAlignment(), readExact(), writeNotApplicable()));
  expectRejected<MemoryAccessClass>(
      "zero lane count",
      MemoryAccessClass::create(
          MemoryAccessForm::Contiguous, singleton(32), singleton(0),
          {MaskInactivePair{MemoryMaskForm::Absent,
                            InactiveLaneSemantics::NotApplicable}},
          scalarAlignment(), readExact(), writeNotApplicable()));
  expectRejected<MemoryAccessClass>(
      "element lane count",
      MemoryAccessClass::create(
          MemoryAccessForm::Element, singleton(32), singleton(2),
          {MaskInactivePair{MemoryMaskForm::Absent,
                            InactiveLaneSemantics::NotApplicable}},
          scalarAlignment(), readExact(), writeNotApplicable()));
}

void checkEnumCodecs() {
  expectRejected<ClosedEnumDomain<ReadSubwordSemantics>>(
      "empty normalized read domain",
      ClosedEnumDomain<ReadSubwordSemantics>::normalize({}));
  require("read tags",
          getCanonicalTag(ReadSubwordSemantics::NotApplicable) == 0 &&
              getCanonicalTag(ReadSubwordSemantics::Exact) == 1 &&
              getCanonicalTag(ReadSubwordSemantics::ZeroExtend) == 2,
          "read-subword tags changed");
  require("write tags",
          getCanonicalTag(WriteSubwordSemantics::NotApplicable) == 0 &&
              getCanonicalTag(WriteSubwordSemantics::Exact) == 1 &&
              getCanonicalTag(WriteSubwordSemantics::ByteEnable) == 2,
          "write-subword tags changed");
  require("inactive tags",
          getCanonicalTag(InactiveLaneSemantics::NotApplicable) == 0 &&
              getCanonicalTag(InactiveLaneSemantics::Suppress) == 1 &&
              getCanonicalTag(InactiveLaneSemantics::SuppressAndZeroFill) == 2,
          "inactive-lane tags changed");
  expectRejected<ReadSubwordSemantics>("unknown read tag",
                                       decodeReadSubwordSemantics(3));
  expectRejected<WriteSubwordSemantics>("unknown write tag",
                                        decodeWriteSubwordSemantics(3));
  expectRejected<InactiveLaneSemantics>("unknown inactive tag",
                                        decodeInactiveLaneSemantics(3));
  expectRejected<ClosedEnumDomain<ReadSubwordSemantics>>(
      "duplicate read domain",
      ClosedEnumDomain<ReadSubwordSemantics>::fromCanonical(
          {ReadSubwordSemantics::Exact, ReadSubwordSemantics::Exact}));
  expectRejected<ClosedEnumDomain<ReadSubwordSemantics>>(
      "unsorted read domain",
      ClosedEnumDomain<ReadSubwordSemantics>::fromCanonical(
          {ReadSubwordSemantics::ZeroExtend, ReadSubwordSemantics::Exact}));

  ClosedEnumDomain<ReadSubwordSemantics> normalized =
      take("read domain normalization",
           ClosedEnumDomain<ReadSubwordSemantics>::normalize(
               {ReadSubwordSemantics::ZeroExtend, ReadSubwordSemantics::Exact,
                ReadSubwordSemantics::ZeroExtend}));
  const ReadSubwordSemantics expected[] = {ReadSubwordSemantics::Exact,
                                           ReadSubwordSemantics::ZeroExtend};
  require("read domain normalization",
          normalized.values() == llvm::ArrayRef(expected),
          "enum values did not normalize by their stable Fabric tags");
  require(
      "read domain lookup",
      normalized.contains(ReadSubwordSemantics::Exact) &&
          normalized.contains(ReadSubwordSemantics::ZeroExtend) &&
          !normalized.contains(ReadSubwordSemantics::NotApplicable) &&
          !normalized.contains(static_cast<ReadSubwordSemantics>(UINT8_MAX)),
      "normalized enum membership is not exact");
  expectRejected<ClosedEnumDomain<ReadSubwordSemantics>>(
      "unknown normalized read domain",
      ClosedEnumDomain<ReadSubwordSemantics>::normalize(
          {static_cast<ReadSubwordSemantics>(UINT8_MAX)}));
}

mlir::func::FuncOp findFunction(mlir::ModuleOp module, llvm::StringRef name) {
  mlir::func::FuncOp found;
  module.walk([&](mlir::func::FuncOp function) {
    if (function.getSymName() == name)
      found = function;
  });
  return found;
}

mlir::Operation *findActor(mlir::func::FuncOp function) {
  mlir::Operation *actor = nullptr;
  function.walk([&](mlir::Operation *operation) {
    if (operation->getName().getDialectNamespace() == "dataflow")
      actor = operation;
  });
  return actor;
}

CanonicalMemoryAccessView accessView(mlir::ModuleOp module,
                                     llvm::StringRef functionName) {
  mlir::func::FuncOp function = findFunction(module, functionName);
  if (!function)
    fail("access fixture", "missing function " + functionName);
  mlir::Operation *actor = findActor(function);
  if (!actor)
    fail("access fixture", "missing actor in " + functionName);
  return take(functionName, getCanonicalMemoryAccessView(actor));
}

void checkTypedAccessMembership(mlir::ModuleOp module) {
  ParameterizedMemoryAccessDomain domain = take(
      "access domain", ParameterizedMemoryAccessDomain::create(
                           {accessClass(MemoryAccessForm::Element, 32, 1),
                            accessClass(MemoryAccessForm::Contiguous, 32, 4)}));
  require("access domain", domain.accessClasses().size() == 2,
          "the canonical access-class range is unavailable");

  expectRejected<ParameterizedMemoryAccessDomain>(
      "overlapping access classes",
      ParameterizedMemoryAccessDomain::create(
          {accessClass(MemoryAccessForm::Element,
                       take("width range", UnsignedDomain::fromCanonical(
                                               {UnsignedInterval{32, 64}})),
                       singleton(1)),
           accessClass(MemoryAccessForm::Element,
                       take("width range", UnsignedDomain::fromCanonical(
                                               {UnsignedInterval{64, 96}})),
                       singleton(1))}));

  CanonicalMemoryAccessView element = accessView(module, "element_f32");
  CanonicalMemoryAccessView vector4 = accessView(module, "vector4_f32");
  CanonicalMemoryAccessView vector2 = accessView(module, "vector2_f64");
  CanonicalMemoryAccessView atomic = accessView(module, "atomic_element_f32");

  require("element membership", domain.contains(element),
          "element f32 was rejected");
  require("vector membership", domain.contains(vector4),
          "contiguous vector<4xf32> was rejected");
  require("equal payload distinction",
          element.dataBits() == 32 && vector4.dataBits() == 128 &&
              vector2.dataBits() == 128,
          "the fixture does not exercise equal vector payload widths");
  require("equal payload distinction", !domain.contains(vector2),
          "contiguous vector<2xf64> collapsed into vector<4xf32>");
  require("atomic alignment mismatch", !domain.contains(atomic),
          "atomic access ignored its exact source alignment");

  ParameterizedMemoryAccessDomain inferredAlignment = take(
      "inferred alignment",
      ParameterizedMemoryAccessDomain::create(
          {take("four-byte access class",
                MemoryAccessClass::create(
                    MemoryAccessForm::Element, singleton(32), singleton(1),
                    {MaskInactivePair{MemoryMaskForm::Absent,
                                      InactiveLaneSemantics::NotApplicable}},
                    fourByteAlignment(), readExact(), writeNotApplicable()))}));
  require("plain alignment derivation", !inferredAlignment.contains(element),
          "plain access alignment was inferred from its type or width");
  require("atomic alignment projection", inferredAlignment.contains(atomic),
          "atomic access did not use its owner-projected source alignment");
}

} // namespace

int main() {
  checkUnsignedDomains();
  checkEnumCodecs();

  mlir::DialectRegistry registry;
  registry
      .insert<DataflowDialect, mlir::func::FuncDialect, mlir::DLTIDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(accessFixture, &context);
  if (!module)
    fail("access fixture", "failed to parse");
  checkTypedAccessMembership(*module);
  return EXIT_SUCCESS;
}
