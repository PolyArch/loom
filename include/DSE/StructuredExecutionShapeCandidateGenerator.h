#ifndef LOOM_DSE_STRUCTUREDEXECUTIONSHAPECANDIDATEGENERATOR_H
#define LOOM_DSE_STRUCTUREDEXECUTIONSHAPECANDIDATEGENERATOR_H

#include "Common/ComponentViewDigest.h"
#include "DSE/CandidateGenerator.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    structuredExecutionShapeCandidateGeneratorKind(3);

class ResolvedStructuredExecutionShapeGeneratorConfigView final {
public:
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedStructuredExecutionShapeGeneratorConfigView(
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedStructuredExecutionShapeGeneratorConfigView>
  projectResolvedStructuredExecutionShapeGeneratorConfigView();
  friend llvm::Expected<ResolvedStructuredExecutionShapeGeneratorConfigView>
  adoptResolvedStructuredExecutionShapeGeneratorConfigView(
      llvm::ArrayRef<std::uint8_t>, llvm::ArrayRef<std::uint8_t>,
      const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t>
resolvedStructuredExecutionShapeGeneratorConfigSchemaBytes();

llvm::Expected<ResolvedStructuredExecutionShapeGeneratorConfigView>
projectResolvedStructuredExecutionShapeGeneratorConfigView();

llvm::Expected<ResolvedStructuredExecutionShapeGeneratorConfigView>
adoptResolvedStructuredExecutionShapeGeneratorConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const CandidateGeneratorDescriptor &
structuredExecutionShapeCandidateGeneratorDescriptor();
llvm::Error registerStructuredExecutionShapeCandidateGenerator();

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindStructuredExecutionShapeCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> structuredPrograms,
    const ArtifactRootReference &fabric);

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveStructuredExecutionShapeCandidateGeneratorBinding(
    const ResolvedStructuredExecutionShapeGeneratorConfigView &config);

} // namespace loom::dse

#endif // LOOM_DSE_STRUCTUREDEXECUTIONSHAPECANDIDATEGENERATOR_H
