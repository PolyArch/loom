#pragma once

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "PnR/System/SystemPnrProblem.h"
#include "SystemCandidateStateTestSupport.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <optional>

namespace loom::pnr::test::fixture {

using loom::pnr::test::byteList;
using loom::pnr::test::bytesAttr;
using loom::pnr::test::unsignedBytes;

[[noreturn]] void fail(const llvm::Twine &message);

void require(bool condition, const llvm::Twine &message);

void requireVerificationFailureContains(mlir::Operation *operation,
                                        llvm::StringRef expected);

::mapping::ArtifactRootReferenceAttr
rootReferenceAttr(mlir::MLIRContext *context,
                  const loom::ArtifactRootReference &reference);

::mapping::SystemServiceObligationKeyAttr
serviceObligationAttr(mlir::MLIRContext *context,
                      const loom::ArtifactIdentity &owner,
                      const loom::mapping::SystemServiceObligationKey &key);

::mapping::SystemTransferTerminalKeyAttr
transferTerminalAttr(mlir::MLIRContext *context,
                     const loom::ArtifactIdentity &owner,
                     const loom::mapping::SystemTransferTerminalKey &key);

mlir::OwningOpRef<mlir::ModuleOp> buildSystemConstraintModule(
    mlir::MLIRContext &context, const loom::ArtifactIdentity &dataflowIdentity,
    const loom::ArtifactIdentity &fabricIdentity,
    llvm::ArrayRef<dataflow::RootThreadLaunchRef> roots);

void addSystemRestriction(mlir::OpBuilder &builder,
                          ::mapping::ConstraintsSystemOp root,
                          ::mapping::SystemConstraintProjection projection,
                          mlir::Attribute subject,
                          llvm::ArrayRef<mlir::Attribute> domain);

void addSystemEquality(mlir::OpBuilder &builder,
                       ::mapping::ConstraintsSystemOp root,
                       ::mapping::SystemConstraintProjection projection,
                       llvm::ArrayRef<mlir::Attribute> subjects);

mlir::MLIRContext makeContext();

dataflow::CanonicalDataflowArtifact buildDataflow(mlir::MLIRContext &context);

dataflow::CanonicalDataflowArtifact
buildCapacityPressureDataflow(mlir::MLIRContext &context);

dataflow::CanonicalDataflowArtifact
buildMemoryDataflow(mlir::MLIRContext &context);

loom::ArtifactRootReference
generateSpatialMapping(const dataflow::CanonicalDataflowProgramView &dataflow,
                       const loom::fabric::FinalizedFabricRoot &module,
                       const loom::ResolvedConfig &resolved,
                       loom::ArtifactStore &store,
                       mlir::MLIRContext *context = nullptr,
                       std::optional<dataflow::GraphRef> cover = std::nullopt);

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void requireFailureContains(llvm::Expected<T> value,
                            llvm::StringRef diagnostic) {
  if (value)
    fail("adverse CandidateState input unexpectedly succeeded");
  const std::string actual = llvm::toString(value.takeError());
  require(llvm::StringRef(actual).contains(diagnostic),
          "adverse diagnostic changed: " + actual);
}

template <typename T>
void requireProvenInfeasibleFreeze(llvm::Expected<T> value,
                                   llvm::StringRef diagnostic) {
  if (value)
    fail("statically infeasible System input unexpectedly froze");
  bool matched = false;
  llvm::Error remaining = llvm::handleErrors(
      value.takeError(), [&](const loom::pnr::SystemPnrFreezeFailure &failure) {
        matched = true;
        require(failure.kind() ==
                    loom::pnr::SystemPnrFreezeFailureKind::ProvenInfeasible,
                "static System failure has the wrong kind");
        std::string actual;
        llvm::raw_string_ostream stream(actual);
        failure.log(stream);
        stream.flush();
        require(llvm::StringRef(actual).contains(diagnostic),
                "static System failure diagnostic changed: " + actual);
      });
  if (remaining)
    fail(llvm::toString(std::move(remaining)));
  require(matched, "static System failure lost its typed cause");
}

template <typename Attr, typename Ref>
Attr constraintDataflowAttr(mlir::MLIRContext *context,
                            const loom::ArtifactIdentity &owner,
                            const Ref &reference) {
  return Attr::get(context,
                   bytesAttr(context, take(dataflow::encodeDataflowReference(
                                          owner, reference))));
}

template <typename Attr, typename Ref>
Attr constraintFabricAttr(mlir::MLIRContext *context, const Ref &reference) {
  return Attr::get(
      context,
      bytesAttr(context, loom::fabric::canonicalFabricBytes(reference)));
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::error_code error = llvm::sys::fs::createUniqueDirectory(
        "loom-system-candidate-state", path_);
    if (error)
      fail("cannot create ArtifactStore directory: " + error.message());
  }

  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }

  llvm::StringRef path() const { return path_; }

private:
  llvm::SmallString<128> path_;
};

} // namespace loom::pnr::test::fixture
