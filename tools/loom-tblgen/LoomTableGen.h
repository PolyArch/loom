#ifndef LOOM_TOOLS_LOOMTBLGEN_LOOMTABLEGEN_H
#define LOOM_TOOLS_LOOMTBLGEN_LOOMTABLEGEN_H

namespace llvm {
class RecordKeeper;
class raw_ostream;
} // namespace llvm

namespace loom {
namespace tblgen {

/// Emits the canonical operation schema rows: the closed semantic vocabulary
/// and every registered actor schema, in numeric-id order.
void emitOperationSchemas(const llvm::RecordKeeper &records,
                          llvm::raw_ostream &os);

/// Emits the implementation-family registry rows: the closed capability
/// parameter and typed admission vocabularies, and every family descriptor
/// with its admitted schema members, in numeric-id order.
void emitImplementationFamilies(const llvm::RecordKeeper &records,
                                llvm::raw_ostream &os);

/// Emits the MLIR enum and specialized attribute declarations for the
/// implementation-family identity.
void emitImplementationFamilyEnum(const llvm::RecordKeeper &records,
                                  llvm::raw_ostream &os);

} // namespace tblgen
} // namespace loom

#endif // LOOM_TOOLS_LOOMTBLGEN_LOOMTABLEGEN_H
