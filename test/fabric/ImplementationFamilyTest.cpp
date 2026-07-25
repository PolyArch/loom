//===- ImplementationFamilyTest.cpp - HSG family registry anchors ---------===//
//
// Anchors the normative implementation-family registry:
//
//   * a family descriptor exposes exactly the four generated facts, and every
//     admitted member is a registered operation schema;
//   * a family admits its own shared members and rejects an operation of
//     another family, so membership is a real relation rather than a name
//     bag; and
//   * the one keyword of a family is derived from its generated identity and
//     round-trips, so a diagnostic never needs a descriptor name field.
//
// The anchor deliberately does not restate the registry: it checks relations
// the generated source must satisfy, plus the few normative memberships the
// hardware-sharing specification fixes by name.
//
//===----------------------------------------------------------------------===//

#include "Fabric/IR/ImplementationFamily.h"

#include "Dataflow/IR/OperationSchema.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace fabric;
using dataflow::OperationSchemaId;

namespace {

bool checkDescriptorRelations() {
  bool ok = true;
  llvm::StringSet<> keywords;
  const std::uint32_t families = implementationFamilyCount();
  const std::uint32_t schemas = dataflow::operationSchemaCount();
  if (families == 0) {
    llvm::errs() << "the registry declares no implementation family\n";
    return false;
  }
  for (std::uint32_t index = 0; index < families; ++index) {
    auto family = static_cast<ImplementationFamilyId>(index);
    const ImplementationFamilyDescriptor &descriptor =
        implementationFamily(family);
    if (descriptor.familyId != family) {
      llvm::errs() << "family " << index << " reports another identity\n";
      ok = false;
    }
    if (descriptor.admittedSchemas.empty()) {
      llvm::errs() << implementationFamilyKeyword(family)
                   << " admits no operation schema\n";
      ok = false;
    }
    for (OperationSchemaId member : descriptor.admittedSchemas) {
      if (static_cast<std::uint32_t>(member) >= schemas) {
        llvm::errs() << implementationFamilyKeyword(family)
                     << " admits an unregistered operation schema\n";
        ok = false;
        continue;
      }
      if (!admitsOperationSchema(family, member)) {
        llvm::errs() << implementationFamilyKeyword(family)
                     << " does not admit its own declared member\n";
        ok = false;
      }
    }

    llvm::StringRef keyword = implementationFamilyKeyword(family);
    if (keyword.empty() || !keywords.insert(keyword).second) {
      llvm::errs() << "family " << index << " has no unique keyword\n";
      ok = false;
    }
    std::optional<ImplementationFamilyId> resolved =
        findImplementationFamily(keyword);
    if (!resolved || *resolved != family) {
      llvm::errs() << "keyword '" << keyword << "' does not resolve back\n";
      ok = false;
    }
    if (capabilityParamsSchemaKeyword(descriptor.capabilityParamsSchema)
            .empty() ||
        typedAdmissionProviderKeyword(descriptor.typedAdmissionProvider)
            .empty()) {
      llvm::errs() << keyword << " selects an unspellable vocabulary member\n";
      ok = false;
    }
  }
  return ok;
}

/// One genuinely shared datapath family admits its own members and nothing
/// else. An adder and a multiplier are separate datapaths, so the add/subtract
/// family must reject a multiply.
bool checkMembershipAndWrongFamily() {
  bool ok = true;
  if (!admitsOperationSchema(ImplementationFamilyId::ScalarIntegerAddSub,
                             OperationSchemaId::ArithAddI) ||
      !admitsOperationSchema(ImplementationFamilyId::ScalarIntegerAddSub,
                             OperationSchemaId::ArithSubI)) {
    llvm::errs() << "the integer add/subtract family lost a shared member\n";
    ok = false;
  }
  if (admitsOperationSchema(ImplementationFamilyId::ScalarIntegerAddSub,
                            OperationSchemaId::ArithMulI)) {
    llvm::errs() << "the integer add/subtract family admitted a multiply\n";
    ok = false;
  }
  if (!admitsOperationSchema(ImplementationFamilyId::ScalarIntegerMultiply,
                             OperationSchemaId::ArithMulI)) {
    llvm::errs() << "the integer multiply family lost its member\n";
    ok = false;
  }

  // The four loop-control families are distinct physical families: none of
  // them admits another's operation.
  const ImplementationFamilyId loopControl[] = {
      ImplementationFamilyId::LoopStream, ImplementationFamilyId::LoopCarry,
      ImplementationFamilyId::LoopInvariant, ImplementationFamilyId::LoopGate};
  const OperationSchemaId loopMembers[] = {
      OperationSchemaId::DataflowStream, OperationSchemaId::DataflowCarry,
      OperationSchemaId::DataflowInvariant, OperationSchemaId::DataflowGate};
  for (unsigned family = 0; family < 4; ++family)
    for (unsigned member = 0; member < 4; ++member) {
      const bool admitted =
          admitsOperationSchema(loopControl[family], loopMembers[member]);
      if (admitted != (family == member)) {
        llvm::errs() << implementationFamilyKeyword(loopControl[family])
                     << " has the wrong relation to "
                     << dataflow::operationSchemaSpelling(loopMembers[member])
                     << '\n';
        ok = false;
      }
    }

  if (findImplementationFamily("NoSuchFamily")) {
    llvm::errs() << "an unregistered keyword resolved to a family\n";
    ok = false;
  }
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok &= checkDescriptorRelations();
  ok &= checkMembershipAndWrongFamily();
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
