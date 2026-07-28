#include "Fabric/IR/MemoryActorContractDomain.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef anchor, llvm::StringRef detail) {
  llvm::errs() << anchor << ": " << detail << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::StringRef anchor, llvm::Expected<T> value) {
  if (!value)
    fail(anchor, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectRejected(llvm::StringRef anchor, llvm::Expected<T> value) {
  if (value)
    fail(anchor, "unexpectedly accepted");
  llvm::consumeError(value.takeError());
}

void require(llvm::StringRef anchor, bool condition, llvm::StringRef detail) {
  if (!condition)
    fail(anchor, detail);
}

dataflow::CanonicalActorSchemaProjection
actor(mlir::MLIRContext &context, dataflow::OperationSchemaId schema,
      dataflow::MemoryContractPayload payload) {
  return {schema, mlir::FunctionType::get(&context, {}, {}),
          dataflow::SemanticPayload(std::move(payload))};
}

dataflow::SyncScopeProjection systemScope() {
  return {dataflow::SyncScopeKind::System, {}, {}};
}

void checkLoadStoreReductionAndCodec(mlir::MLIRContext &context) {
  fabric::MemoryActorContractClause plainFalse =
      fabric::LoadStorePlainContractClause{{false}};
  fabric::MemoryActorContractClause plainTrue =
      fabric::LoadStorePlainContractClause{{true}};

  fabric::MemoryActorContractDomain domain =
      take("plain reduction", fabric::MemoryActorContractDomain::create(
                                  dataflow::OperationSchemaId::DataflowLoad,
                                  {plainTrue, plainFalse}));
  require("plain reduction", domain.clauses().size() == 1,
          "equivalent plain clauses were not reduced");

  require(
      "plain membership",
      domain.contains(actor(context, dataflow::OperationSchemaId::DataflowLoad,
                            dataflow::PlainAccessProjection{true})),
      "accepted volatile load contract was rejected");
  require("schema membership",
          !domain.contains(actor(context,
                                 dataflow::OperationSchemaId::DataflowStore,
                                 dataflow::PlainAccessProjection{true})),
          "foreign actor schema was accepted");

  expectRejected<fabric::MemoryActorContractDomain>(
      "noncanonical split",
      fabric::MemoryActorContractDomain::fromCanonical(
          dataflow::OperationSchemaId::DataflowLoad, {plainFalse, plainTrue}));
  expectRejected<fabric::MemoryActorContractDomain>(
      "wrong clause variant",
      fabric::MemoryActorContractDomain::create(
          dataflow::OperationSchemaId::DataflowAtomicRmw, {plainFalse}));

  std::vector<std::uint8_t> bytes = take(
      "actor domain encode", fabric::encodeMemoryActorContractDomain(domain));
  fabric::MemoryActorContractDomain decoded =
      take("actor domain decode",
           fabric::decodeMemoryActorContractDomain(bytes, &context));
  std::vector<std::uint8_t> rewritten = take(
      "actor domain rewrite", fabric::encodeMemoryActorContractDomain(decoded));
  require("actor domain codec", bytes == rewritten,
          "strict actor-domain round trip changed canonical bytes");
}

void checkCompareExchangePairCorrelation(mlir::MLIRContext &context) {
  fabric::CompareExchangeContractClause clause{
      {{dataflow::AtomicOrdering::Acquire, dataflow::AtomicOrdering::Monotonic},
       {dataflow::AtomicOrdering::SeqCst, dataflow::AtomicOrdering::Acquire}},
      {systemScope()},
      {std::nullopt},
      {false},
      {false}};
  fabric::MemoryActorContractDomain domain =
      take("compare-exchange domain",
           fabric::MemoryActorContractDomain::create(
               dataflow::OperationSchemaId::DataflowCmpXchg,
               {fabric::MemoryActorContractClause(clause)}));

  dataflow::CompareExchangeProjection accepted{
      dataflow::AtomicOrdering::SeqCst,
      dataflow::AtomicOrdering::Acquire,
      systemScope(),
      8,
      std::nullopt,
      false,
      false};
  dataflow::CompareExchangeProjection crossProduct = accepted;
  crossProduct.successOrdering = dataflow::AtomicOrdering::Acquire;
  require("ordering-pair membership",
          domain.contains(actor(
              context, dataflow::OperationSchemaId::DataflowCmpXchg, accepted)),
          "declared compare-exchange ordering pair was rejected");
  require("ordering-pair correlation",
          !domain.contains(actor(context,
                                 dataflow::OperationSchemaId::DataflowCmpXchg,
                                 crossProduct)),
          "ordering pairs expanded into a Cartesian product");
}

} // namespace

int main() {
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  checkLoadStoreReductionAndCodec(context);
  checkCompareExchangePairCorrelation(context);
  return EXIT_SUCCESS;
}
