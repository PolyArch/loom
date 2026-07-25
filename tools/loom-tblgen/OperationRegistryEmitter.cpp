//===- OperationRegistryEmitter.cpp - Operation and family registry -------===//
//
// Emits the canonical operation registry and the implementation-family
// registry from `include/Dataflow/IR/OperationSchemas.td`.
//
// Both outputs are X-macro row files. A row carries only the facts its
// declaration owns: the record name is the C++ enumerator and, for a family,
// the one attribute keyword. A schema's stable spelling is never restated
// here; the emitted row names the operation class and the reader expands
// `OpClass::getOperationName()`.
//
// Generation fails closed on a duplicate id or record name, on an actor kind
// or semantic case outside its closed C++ enum, on an empty or repeated
// family member list, and on any id domain that is not dense `[0, N)`. Rows
// are emitted in numeric-id order, so the bytes depend only on the source
// records and never on record-map iteration order.
//
//===----------------------------------------------------------------------===//

#include "LoomTableGen.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

using namespace llvm;

namespace {

/// The closed C++ enums a row may name. They are declared in
/// `Dataflow/IR/OperationSchema.h`; naming anything else would emit a row that
/// does not compile, so it is rejected at generation instead.
constexpr StringLiteral kActorKinds[] = {"Compute", "Control", "Memory"};

struct SchemaRow {
  int64_t id;
  StringRef name;
  StringRef opClass;
  StringRef actorKind;
  StringRef semanticsCase;
};

struct FamilyRow {
  int64_t id;
  StringRef name;
  StringRef capabilityParams;
  StringRef typedAdmission;
  SmallVector<StringRef, 8> members;
};

struct VocabularyRow {
  int64_t id;
  StringRef name;
};

/// Every non-anonymous record deriving from `superclass`.
std::vector<const Record *> recordsOf(const RecordKeeper &records,
                                      StringRef superclass) {
  std::vector<const Record *> found;
  for (const Record *record : records.getAllDerivedDefinitions(superclass))
    if (!record->isAnonymous())
      found.push_back(record);
  if (found.empty())
    PrintFatalError("the source declares no " + superclass + " record");
  return found;
}

/// Sorts by numeric id and requires one dense `[0, N)` domain with unique
/// record names, so the emitted table is a dense array indexed by id.
template <typename Row>
void requireDenseDomain(std::vector<Row> &rows, StringRef domain) {
  llvm::sort(rows,
             [](const Row &lhs, const Row &rhs) { return lhs.id < rhs.id; });
  StringSet<> names;
  for (auto [index, row] : llvm::enumerate(rows)) {
    if (row.id != static_cast<int64_t>(index))
      PrintFatalError(domain + " ids must form a dense [0, N) domain; id " +
                      Twine(row.id) + " is out of place at index " +
                      Twine(static_cast<int64_t>(index)));
    if (!names.insert(row.name).second)
      PrintFatalError("duplicate " + domain + " record name " + row.name);
  }
}

std::vector<VocabularyRow> readVocabulary(const RecordKeeper &records,
                                          StringRef superclass,
                                          StringRef idField, StringRef domain) {
  std::vector<VocabularyRow> rows;
  for (const Record *record : recordsOf(records, superclass))
    rows.push_back({record->getValueAsInt(idField), record->getName()});
  requireDenseDomain(rows, domain);
  return rows;
}

std::vector<SchemaRow> readSchemas(const RecordKeeper &records,
                                   const std::vector<VocabularyRow> &cases) {
  StringSet<> caseNames;
  for (const VocabularyRow &row : cases)
    caseNames.insert(row.name);

  std::vector<SchemaRow> rows;
  for (const Record *record : recordsOf(records, "ActorSchema")) {
    StringRef actorKind = record->getValueAsString("actorKind");
    if (!llvm::is_contained(kActorKinds, actorKind))
      PrintFatalError(record->getName() + " names actor kind '" + actorKind +
                      "', which is not a declared canonical actor kind");
    StringRef semantics = record->getValueAsDef("semantics")->getName();
    if (!caseNames.contains(semantics))
      PrintFatalError(record->getName() + " names semantics case '" +
                      semantics + "', which is not declared");
    rows.push_back({record->getValueAsInt("schemaId"), record->getName(),
                    record->getValueAsString("opClass"), actorKind, semantics});
  }
  requireDenseDomain(rows, "operation schema");

  StringSet<> classes;
  for (const SchemaRow &row : rows)
    if (!classes.insert(row.opClass).second)
      PrintFatalError("two operation schemas name the same operation class " +
                      row.opClass);
  return rows;
}

std::vector<FamilyRow> readFamilies(const RecordKeeper &records,
                                    const std::vector<SchemaRow> &schemas) {
  StringSet<> schemaNames;
  for (const SchemaRow &row : schemas)
    schemaNames.insert(row.name);

  std::vector<FamilyRow> rows;
  for (const Record *record : recordsOf(records, "ImplementationFamily")) {
    FamilyRow row{record->getValueAsInt("familyId"),
                  record->getName(),
                  record->getValueAsDef("capabilityParams")->getName(),
                  record->getValueAsDef("typedAdmission")->getName(),
                  {}};
    StringSet<> seen;
    for (const Record *member :
         record->getValueAsListOfDefs("admittedSchemas")) {
      if (!schemaNames.contains(member->getName()))
        PrintFatalError(record->getName() + " admits '" + member->getName() +
                        "', which is not a registered operation schema");
      if (!seen.insert(member->getName()).second)
        PrintFatalError(record->getName() + " repeats member " +
                        member->getName());
      row.members.push_back(member->getName());
    }
    if (row.members.empty())
      PrintFatalError(record->getName() + " admits no operation schema");
    rows.push_back(std::move(row));
  }
  requireDenseDomain(rows, "implementation family");
  return rows;
}

void emitHeader(raw_ostream &os, StringRef what) {
  os << "//===- Generated by loom-tblgen from OperationSchemas.td "
        "-----*- C++ -*-===//\n"
     << "//\n"
     << "// " << what << "\n"
     << "// Do not edit. Edit include/Dataflow/IR/OperationSchemas.td.\n"
     << "//\n"
     << "//===------------------------------------------------------------"
        "----------===//\n\n";
}

void emitTableGenHeader(raw_ostream &os, StringRef what) {
  os << "//===- Generated by loom-tblgen from OperationSchemas.td "
        "---*- tablegen -*-===//\n"
     << "//\n"
     << "// " << what << "\n"
     << "// Do not edit. Edit include/Dataflow/IR/OperationSchemas.td.\n"
     << "//\n"
     << "//===------------------------------------------------------------"
        "----------===//\n\n";
}

/// Emits `#ifndef Name\n#define Name(...)\n#endif` guards so a consumer
/// expands only the rows it cares about.
void emitMacroGuard(raw_ostream &os, StringRef macro, StringRef params) {
  os << "#ifndef " << macro << "\n"
     << "#define " << macro << "(" << params << ")\n"
     << "#endif\n";
}

void emitMacroUndef(raw_ostream &os, StringRef macro) {
  os << "#undef " << macro << "\n";
}

} // namespace

void loom::tblgen::emitOperationSchemas(const RecordKeeper &records,
                                        raw_ostream &os) {
  std::vector<VocabularyRow> cases =
      readVocabulary(records, "SemanticsCase", "caseId", "semantics case");
  std::vector<SchemaRow> schemas = readSchemas(records, cases);

  emitHeader(os, "The canonical operation schema rows.");
  emitMacroGuard(os, "LOOM_OPERATION_SEMANTICS_CASE", "Name, Id");
  emitMacroGuard(os, "LOOM_OPERATION_SCHEMA",
                 "Name, Id, OpClass, ActorKind, SemanticsCase");
  os << "\n";

  for (const VocabularyRow &row : cases)
    os << "LOOM_OPERATION_SEMANTICS_CASE(" << row.name << ", " << row.id
       << ")\n";
  os << "\n";
  for (const SchemaRow &row : schemas)
    os << "LOOM_OPERATION_SCHEMA(" << row.name << ", " << row.id << ", "
       << row.opClass << ", " << row.actorKind << ", " << row.semanticsCase
       << ")\n";

  os << "\n";
  emitMacroUndef(os, "LOOM_OPERATION_SEMANTICS_CASE");
  emitMacroUndef(os, "LOOM_OPERATION_SCHEMA");
}

void loom::tblgen::emitImplementationFamilies(const RecordKeeper &records,
                                              raw_ostream &os) {
  std::vector<VocabularyRow> cases =
      readVocabulary(records, "SemanticsCase", "caseId", "semantics case");
  std::vector<SchemaRow> schemas = readSchemas(records, cases);
  std::vector<VocabularyRow> params = readVocabulary(
      records, "CapabilityParamsSchema", "paramsId", "capability params");
  std::vector<VocabularyRow> providers = readVocabulary(
      records, "TypedAdmissionProvider", "providerId", "admission provider");
  std::vector<FamilyRow> families = readFamilies(records, schemas);

  emitHeader(os, "The normative implementation-family registry rows.");
  emitMacroGuard(os, "LOOM_CAPABILITY_PARAMS_SCHEMA", "Name, Id");
  emitMacroGuard(os, "LOOM_TYPED_ADMISSION_PROVIDER", "Name, Id");
  emitMacroGuard(os, "LOOM_IMPLEMENTATION_FAMILY",
                 "Name, Id, CapabilityParams, TypedAdmission");
  emitMacroGuard(os, "LOOM_IMPLEMENTATION_FAMILY_MEMBER", "Family, Schema");
  emitMacroGuard(os, "LOOM_IMPLEMENTATION_FAMILY_END", "Name");
  os << "\n";

  for (const VocabularyRow &row : params)
    os << "LOOM_CAPABILITY_PARAMS_SCHEMA(" << row.name << ", " << row.id
       << ")\n";
  os << "\n";
  for (const VocabularyRow &row : providers)
    os << "LOOM_TYPED_ADMISSION_PROVIDER(" << row.name << ", " << row.id
       << ")\n";
  os << "\n";
  for (const FamilyRow &row : families) {
    os << "LOOM_IMPLEMENTATION_FAMILY(" << row.name << ", " << row.id << ", "
       << row.capabilityParams << ", " << row.typedAdmission << ")\n";
    for (StringRef member : row.members)
      os << "LOOM_IMPLEMENTATION_FAMILY_MEMBER(" << row.name << ", " << member
         << ")\n";
    os << "LOOM_IMPLEMENTATION_FAMILY_END(" << row.name << ")\n";
  }

  os << "\n";
  emitMacroUndef(os, "LOOM_CAPABILITY_PARAMS_SCHEMA");
  emitMacroUndef(os, "LOOM_TYPED_ADMISSION_PROVIDER");
  emitMacroUndef(os, "LOOM_IMPLEMENTATION_FAMILY");
  emitMacroUndef(os, "LOOM_IMPLEMENTATION_FAMILY_MEMBER");
  emitMacroUndef(os, "LOOM_IMPLEMENTATION_FAMILY_END");
}

void loom::tblgen::emitImplementationFamilyEnum(const RecordKeeper &records,
                                                raw_ostream &os) {
  std::vector<VocabularyRow> cases =
      readVocabulary(records, "SemanticsCase", "caseId", "semantics case");
  std::vector<SchemaRow> schemas = readSchemas(records, cases);
  std::vector<FamilyRow> families = readFamilies(records, schemas);

  emitTableGenHeader(os, "The implementation-family MLIR enum attribute.");
  for (const FamilyRow &row : families)
    os << "def Fabric_ImplementationFamily_" << row.name
       << " : I32EnumAttrCase<\"" << row.name << "\", " << row.id << ", \""
       << row.name << "\">;\n";

  os << "\ndef Fabric_ImplementationFamilyId : I32EnumAttr<\n"
     << "    \"ImplementationFamilyId\",\n"
     << "    \"registered Hardware Sharing Group implementation family\",\n"
     << "    [\n";
  for (auto [index, row] : llvm::enumerate(families)) {
    os << "      Fabric_ImplementationFamily_" << row.name;
    if (index + 1 != families.size())
      os << ',';
    os << "\n";
  }
  os << "    ]> {\n"
     << "  let cppNamespace = \"::fabric\";\n"
     << "  let genSpecializedAttr = 0;\n"
     << "}\n\n"
     << "def Fabric_ImplementationFamilyAttr\n"
     << "    : EnumAttr<Fabric_Dialect, Fabric_ImplementationFamilyId,\n"
     << "               \"implementation_family\"> {\n"
     << "  let assemblyFormat = \"`<` $value `>`\";\n"
     << "}\n";
}
