#ifndef FABRIC_IR_IMPLEMENTATIONFAMILY_H
#define FABRIC_IR_IMPLEMENTATIONFAMILY_H

#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/FabricEnums.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace fabric {

/// The closed typed `hw_params` record schema a family selects.
enum class CapabilityParamsSchemaId : std::uint32_t {
#define LOOM_CAPABILITY_PARAMS_SCHEMA(Name, Id) Name = Id,
#include "Fabric/IR/ImplementationFamilies.inc"
};

/// The closed typed admission rule a family selects.
enum class TypedAdmissionProviderId : std::uint32_t {
#define LOOM_TYPED_ADMISSION_PROVIDER(Name, Id) Name = Id,
#include "Fabric/IR/ImplementationFamilies.inc"
};

/// The one normative family descriptor. It owns exactly four facts: the stable
/// family identity, the admitted registered operation schemas, the closed
/// typed capability-parameter record schema, and the typed admission provider.
///
/// It carries no name, spelling, shape policy, port shape, state, timing, or
/// backend field. Diagnostic spelling is derived from the family identity.
struct ImplementationFamilyDescriptor {
  ImplementationFamilyId familyId;
  llvm::ArrayRef<::dataflow::OperationSchemaId> admittedSchemas;
  CapabilityParamsSchemaId capabilityParamsSchema;
  TypedAdmissionProviderId typedAdmissionProvider;
};

/// Count of registered families. Every family id is in `[0, count)`.
std::uint32_t implementationFamilyCount();

/// The descriptor of one registered family. Lookup is a dense index.
const ImplementationFamilyDescriptor &
implementationFamily(ImplementationFamilyId family);

/// The one keyword of a family, derived from its generated identity. It is the
/// only spelling the typed attribute accepts and prints, and the only spelling
/// a diagnostic uses.
llvm::StringRef implementationFamilyKeyword(ImplementationFamilyId family);

/// The family named by `keyword`, or absent when none is.
std::optional<ImplementationFamilyId>
findImplementationFamily(llvm::StringRef keyword);

/// Whether `family` admits `schema`. The admitted set of a real shared
/// datapath family is small and fixed, so this is a bounded scan of the one
/// generated relation.
bool admitsOperationSchema(ImplementationFamilyId family,
                           ::dataflow::OperationSchemaId schema);

/// Diagnostic spellings of the two closed vocabularies a descriptor selects.
llvm::StringRef capabilityParamsSchemaKeyword(CapabilityParamsSchemaId schema);
llvm::StringRef
typedAdmissionProviderKeyword(TypedAdmissionProviderId provider);

} // namespace fabric

#endif // FABRIC_IR_IMPLEMENTATIONFAMILY_H
