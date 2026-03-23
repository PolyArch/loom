//===-- ADGBuilderNoCPorts.cpp - NoC port extensions for ADGBuilder -*- C++ -*-//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Implements multi-core extension methods on ADGBuilder: NoC ingress/egress
// port creation, SPM capacity setting, and core type export.
//
//===----------------------------------------------------------------------===//

#include "loom/ADG/ADGBuilderDetail.h"

#include "llvm/Support/ErrorHandling.h"

#include <sstream>

namespace loom {
namespace adg {

using namespace detail;

unsigned ADGBuilder::addNoCIngressPort(const std::string &name,
                                       unsigned bitsWidth) {
  unsigned portIdx = addScalarInput(name, bitsWidth);
  impl_->nocPorts.push_back({name, bitsWidth, /*isIngress=*/true});
  return portIdx;
}

unsigned ADGBuilder::addNoCEgressPort(const std::string &name,
                                      unsigned bitsWidth) {
  unsigned portIdx = addScalarOutput(name, bitsWidth);
  impl_->nocPorts.push_back({name, bitsWidth, /*isIngress=*/false});
  return portIdx;
}

void ADGBuilder::setSPMCapacity(uint64_t bytes) {
  impl_->spmCapacityBytes = bytes;
  impl_->hasSPMCapacity = true;
}

std::string ADGBuilder::exportCoreType(const std::string &typeName) {
  impl_->coreTypeName = typeName;

  // Generate the MLIR text with the given type name as the module name.
  // We temporarily swap the module name to the type name for export.
  std::string origName = impl_->moduleName;
  impl_->moduleName = typeName;

  std::string mlirText = impl_->generateMLIR("");

  // Restore original module name
  impl_->moduleName = origName;

  // Inject multi-core attributes into the fabric.module line.
  // We need to add noc_port markers and spm_capacity_bytes.
  // The attributes are added to the MLIR text as module-level attributes.
  //
  // For now, we embed them as comments that SystemADGBuilder can parse,
  // plus we store the NoC port names in the metadata section.
  //
  // The actual attribute injection is handled by appending a metadata
  // comment block at the end.
  std::ostringstream metadata;
  metadata << "// CORE_TYPE_METADATA\n";

  if (impl_->hasSPMCapacity) {
    metadata << "// spm_capacity_bytes = " << impl_->spmCapacityBytes << "\n";
  }

  for (const auto &port : impl_->nocPorts) {
    metadata << "// noc_port: " << port.name << " "
             << (port.isIngress ? "ingress" : "egress") << " " << port.bitsWidth
             << "\n";
  }

  return mlirText + metadata.str();
}

} // namespace adg
} // namespace loom
