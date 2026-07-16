#include "MappingHardware.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/Twine.h"

#include <utility>

namespace loom::pnr::detail {

namespace {

std::uint64_t integerAttrValue(mlir::Attribute attr) {
  if (auto intAttr = llvm::dyn_cast_if_present<mlir::IntegerAttr>(attr))
    return static_cast<std::uint64_t>(intAttr.getInt());
  return 0;
}

bool isNamedFabricTemplate(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  if (name != "fabric.pe" && name != "fabric.fu" && name != "fabric.switch" &&
      name != "fabric.mem")
    return false;
  return op->hasAttr("sym_name");
}

std::pair<std::uint64_t, std::uint64_t> memPortCounts(mlir::Operation *op) {
  std::uint64_t loadPorts = 0;
  std::uint64_t storePorts = 0;
  auto hwParams = op->getAttrOfType<mlir::ArrayAttr>("hw_params");
  if (hwParams && !hwParams.empty()) {
    if (auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(hwParams[0])) {
      loadPorts = integerAttrValue(dict.get("load_group_size"));
      storePorts = integerAttrValue(dict.get("store_group_size"));
    }
  }
  return {loadPorts, storePorts};
}

} // namespace

bool isConcreteHardwareOperation(mlir::Operation *op,
                                 mlir::Operation *hardwareRoot) {
  for (mlir::Operation *current = op; current && current != hardwareRoot;
       current = current->getParentOp()) {
    if (isNamedFabricTemplate(current))
      return false;
    if (current->getName().getStringRef() == "fabric.module")
      return false;
  }
  return true;
}

llvm::SmallVector<ConcreteMemOccurrence, 2>
collectConcreteMemOccurrences(mlir::Operation *hardwareRoot) {
  llvm::SmallVector<ConcreteMemOccurrence, 2> occurrences;
  std::uint64_t nextLoadResource = 0;
  std::uint64_t nextStoreResource = 0;
  hardwareRoot->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() != "fabric.mem" ||
        !isConcreteHardwareOperation(op, hardwareRoot))
      return;
    auto [loadCount, storeCount] = memPortCounts(op);
    occurrences.push_back(ConcreteMemOccurrence{
        op, MemOccurrenceIdentity{nextLoadResource, loadCount,
                                  nextStoreResource, storeCount}});
    nextLoadResource += loadCount;
    nextStoreResource += storeCount;
  });
  return occurrences;
}

std::string memResourceId(llvm::StringRef hardwareName, MemAccessKind kind,
                          const MemOccurrenceIdentity &identity,
                          std::uint64_t portIndex) {
  llvm::StringRef access = kind == MemAccessKind::Load ? "load" : "store";
  std::uint64_t base = kind == MemAccessKind::Load ? identity.loadResourceBase
                                                   : identity.storeResourceBase;
  return (hardwareName + "::mem." + access + "#" +
          llvm::Twine(base + portIndex))
      .str();
}

} // namespace loom::pnr::detail
