#ifndef LOOM_ADG_BUILDER_INTERNAL_H
#define LOOM_ADG_BUILDER_INTERNAL_H

#include "ADG/Builder.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace loom::adg {

struct ModuleBuilderInternals {
  using BodyLineSpec = ModuleBuilder::BodyLineSpec;
  using BodyOpSpec = ModuleBuilder::BodyOpSpec;
  using BodyResultSpec = ModuleBuilder::BodyResultSpec;

  static ModuleBuilder &addBodyOp(ModuleBuilder &builder, BodyOpSpec op) {
    return builder.addBodyOp(std::move(op));
  }

  static ModuleBuilder &addUnsupportedResource(ModuleBuilder &builder,
                                               std::string detail) {
    return builder.addUnsupportedResource(std::move(detail));
  }
};

namespace detail {

using BodyLineSpec = ModuleBuilderInternals::BodyLineSpec;
using BodyOpSpec = ModuleBuilderInternals::BodyOpSpec;
using BodyResultSpec = ModuleBuilderInternals::BodyResultSpec;

inline ModuleBuilder &recordUnsupportedCatalogResource(
    ModuleBuilder &builder,
    std::initializer_list<::dataflow::OperationSchemaId> schemas) {
  std::string detail;
  llvm::raw_string_ostream os(detail);
  bool first = true;
  for (::dataflow::OperationSchemaId schema : schemas) {
    if (!first)
      os << ", ";
    first = false;
    os << ::dataflow::operationSchemaSpelling(schema);
  }
  os << " has no registered implementation family";
  return ModuleBuilderInternals::addUnsupportedResource(builder, os.str());
}

struct VisualPoint {
  llvm::StringRef node;
  int x;
  int y;
};

PeSpec makeMinimalAddPe(Schedule schedule, std::string lhsSource,
                        std::string rhsSource, std::string boundaryType,
                        std::string fuType,
                        TemporalPeConfig temporal = TemporalPeConfig());
PeSpec makeMinimalAddPe(Schedule schedule, std::string boundaryType,
                        std::string fuType,
                        TemporalPeConfig temporal = TemporalPeConfig());
void addVisualLayout(ModuleBuilder &module, llvm::ArrayRef<VisualPoint> points);

std::vector<std::string> axiManagerPort(std::string port);
std::vector<std::string> axiSubordinatePort(std::string port);
void appendPorts(std::vector<std::string> &dst, std::vector<std::string> src);
void connectAxiMemoryPort(SystemBuilder &system, llvm::StringRef managerNode,
                          llvm::StringRef managerPort,
                          llvm::StringRef memoryNode,
                          llvm::StringRef memoryPort);

ModuleBuilder
makeTopologyMatrixModule(llvm::StringRef name, bool includeTemporal = false,
                         llvm::ArrayRef<VisualPoint> visualPoints = {});

BodyLineSpec exactBodyLine(std::string text);
BodyLineSpec nestedBodyLine(std::string text);
BodyLineSpec directBodyLine(std::vector<std::string> fragments,
                            std::vector<std::string> operands);
ModuleBuilder &appendBodyOp(ModuleBuilder &module, BodyOpSpec op);
std::string bodyResultTypes(llvm::ArrayRef<BodyResultSpec> results);
BodyOpSpec bodyOpWithResultLine(std::vector<BodyResultSpec> results,
                                std::vector<BodyLineSpec> lines,
                                llvm::StringRef prefix,
                                llvm::StringRef suffix = "");
BodyLineSpec directOperandListLine(std::string prefix,
                                   llvm::ArrayRef<std::string> operands,
                                   std::string suffix = "",
                                   llvm::StringRef separator = ", ");
BodyLineSpec directOperandListLine(std::string prefix,
                                   llvm::ArrayRef<llvm::StringRef> operands,
                                   std::string suffix = "",
                                   llvm::StringRef separator = ", ");
BodyLineSpec directOperandListLine(
    std::string prefix, std::initializer_list<llvm::StringRef> operands,
    std::string suffix = "", llvm::StringRef separator = ", ");
BodyLineSpec directHeadAndListLine(std::string prefix, std::string head,
                                   std::string infix,
                                   llvm::ArrayRef<std::string> operands,
                                   std::string suffix);

void addUniformSwitch(ModuleBuilder &module,
                      llvm::ArrayRef<std::string> results,
                      llvm::ArrayRef<std::string> inputs, llvm::StringRef type);
void addUniformSwitch(ModuleBuilder &module,
                      std::initializer_list<llvm::StringRef> results,
                      std::initializer_list<llvm::StringRef> inputs,
                      llvm::StringRef type);
void addUniformSwitch(ModuleBuilder &module,
                      std::initializer_list<llvm::StringRef> results,
                      llvm::ArrayRef<std::string> inputs, llvm::StringRef type);
void addSpatialMemLoad(ModuleBuilder &module);
void addSpatialSwitch(ModuleBuilder &module,
                      llvm::ArrayRef<llvm::StringRef> results,
                      llvm::ArrayRef<llvm::StringRef> inputs,
                      llvm::ArrayRef<llvm::StringRef> rows);

/// The builtin catalog selects its concrete typed domains once here. These
/// helpers are not public Builder defaults: custom ADGs construct an explicit
/// FabricOpCapability with their own typed value.
FabricOpCapability builtinOpCapability(::fabric::ImplementationFamilyId family);
FabricOpCapability
builtinIndexCastCapability(::fabric::ResolvedIndexWidth resolvedIndexWidth);
FabricOpCapability loopStreamCapability(
    ::dataflow::StreamStepKind stepKind,
    std::initializer_list<::mlir::arith::CmpIPredicate> predicates);

llvm::Error validateFabricOpCapability(const FabricOpCapability &capability);
void printFabricOpAttrs(llvm::raw_ostream &os, ::mlir::MLIRContext &context,
                        const FabricOpCapability &capability);
std::string fabricOpAttrsText(const FabricOpCapability &capability);

void addSpatialAddPe(ModuleBuilder &module, llvm::StringRef result,
                     llvm::StringRef lhs, llvm::StringRef rhs,
                     ::fabric::ImplementationFamilyId family,
                     ::dataflow::OperationSchemaId member);
void addUnaryPe(ModuleBuilder &module, llvm::StringRef result,
                llvm::StringRef input, ::fabric::ImplementationFamilyId family,
                ::dataflow::OperationSchemaId member);
void addWideExtensionPe(ModuleBuilder &module, llvm::StringRef result,
                        llvm::StringRef input,
                        ::fabric::ImplementationFamilyId family,
                        ::dataflow::OperationSchemaId member);
void addWideNarrowingPe(ModuleBuilder &module, llvm::StringRef result,
                        llvm::StringRef input,
                        ::fabric::ImplementationFamilyId family,
                        ::dataflow::OperationSchemaId member);
void addWideTruncPe(ModuleBuilder &module, llvm::StringRef result,
                    llvm::StringRef input);
void addTernaryPe(ModuleBuilder &module, llvm::StringRef result,
                  llvm::StringRef lhs, llvm::StringRef rhs, llvm::StringRef acc,
                  ::fabric::ImplementationFamilyId family,
                  ::dataflow::OperationSchemaId member);
std::string numbered(llvm::StringRef prefix, unsigned index);
void addConfigurableConstantPe(ModuleBuilder &module, llvm::StringRef result,
                               llvm::StringRef control,
                               llvm::ArrayRef<llvm::StringRef> constHexValues);
void addConfigurableWideConstantPe(
    ModuleBuilder &module, llvm::StringRef result, llvm::StringRef control,
    llvm::ArrayRef<llvm::StringRef> constHexValues);
void addConfigurableBinaryPe(
    ModuleBuilder &module, llvm::StringRef result, llvm::StringRef lhs,
    llvm::StringRef rhs, ::fabric::ImplementationFamilyId family,
    llvm::ArrayRef<::dataflow::OperationSchemaId> members);
void addConfigurableWideBinaryPe(
    ModuleBuilder &module, llvm::StringRef result, llvm::StringRef lhs,
    llvm::StringRef rhs, ::fabric::ImplementationFamilyId family,
    llvm::ArrayRef<::dataflow::OperationSchemaId> members);
void addCmpPe(ModuleBuilder &module, llvm::StringRef result,
              llvm::StringRef lhs, llvm::StringRef rhs);
void addWideCmpPe(ModuleBuilder &module, llvm::StringRef result,
                  llvm::StringRef lhs, llvm::StringRef rhs);
void addFloatCmpPe(ModuleBuilder &module, llvm::StringRef result,
                   llvm::StringRef lhs, llvm::StringRef rhs);
void addControlSyncPe(ModuleBuilder &module, llvm::StringRef prefix,
                      unsigned inputCount);
void addSelectPe(ModuleBuilder &module, llvm::StringRef result,
                 llvm::StringRef pred, llvm::StringRef trueValue,
                 llvm::StringRef falseValue);
void addWideSelectPe(ModuleBuilder &module, llvm::StringRef result,
                     llvm::StringRef pred, llvm::StringRef trueValue,
                     llvm::StringRef falseValue);
void addDataMuxPe(ModuleBuilder &module, llvm::StringRef result,
                  llvm::StringRef pred, llvm::StringRef falseValue,
                  llvm::StringRef trueValue);
void addWideDataMuxPe(ModuleBuilder &module, llvm::StringRef result,
                      llvm::StringRef pred, llvm::StringRef falseValue,
                      llvm::StringRef trueValue);
void addControlMuxPe(ModuleBuilder &module, llvm::StringRef result,
                     llvm::StringRef pred, llvm::StringRef falseValue,
                     llvm::StringRef trueValue);
void addDataDemuxPe(ModuleBuilder &module, llvm::StringRef falseResult,
                    llvm::StringRef trueResult, llvm::StringRef pred,
                    llvm::StringRef value);
void addWideDataDemuxPe(ModuleBuilder &module, llvm::StringRef falseResult,
                        llvm::StringRef trueResult, llvm::StringRef pred,
                        llvm::StringRef value);
void addControlDemuxPe(ModuleBuilder &module, llvm::StringRef falseResult,
                       llvm::StringRef trueResult, llvm::StringRef pred,
                       llvm::StringRef value);
void addMemoryReductionMem(ModuleBuilder &module, unsigned loadCount,
                           unsigned storeCount);
void addTwoLoadOneStoreMem(ModuleBuilder &module);

ModuleBuilder buildChain1DAdg();
ModuleBuilder buildMesh2DAdg();
ModuleBuilder buildTorusEdgeAdg();
ModuleBuilder buildSystolicArrayAdg();
ModuleBuilder buildClusteredArrayAdg();
ModuleBuilder buildFoldedRingAdg();
ModuleBuilder buildMeshDiagonalAdg();
ModuleBuilder buildMultiLanePipelineAdg();
ModuleBuilder buildReductionTreeAdg();
ModuleBuilder buildCrossCoupledSwitchAdg();
ModuleBuilder buildDiamondBypassAdg();
ModuleBuilder buildMemoryFanoutAdg();
ModuleBuilder buildMixedTemporalBridgeAdg();
ModuleBuilder buildSparseLongLinkAdg();
ModuleBuilder buildHeterogeneousIslandsAdg();

SystemBuilder buildDualSpatialSharedMemorySocAdg();
SystemBuilder buildCachedDualAccelSocAdg();
SystemBuilder buildDmaScratchpadSocAdg();
SystemBuilder buildFixedAndSpatialSocAdg();
SystemBuilder buildTriSpatialSharedMemorySocAdg();
SystemBuilder buildDualHostSharedMemorySocAdg();
SystemBuilder buildPrivateScratchpadPairSocAdg();
SystemBuilder buildHostCacheDualMemorySocAdg();
SystemBuilder buildDmaDualMemorySocAdg();
SystemBuilder buildCachedAcceleratorClusterSocAdg();
SystemBuilder buildMixedFixedSpatialPipelineSocAdg();
SystemBuilder buildSignalQuantizedPairSocAdg();

llvm::Error printReusableSpatialTemplates(llvm::raw_ostream &os,
                                          bool includeVectorAlu,
                                          bool includeMemoryReduction = false,
                                          bool includeSignalWindow = false,
                                          bool includeQuantizedWindow = false);

void addSharedReductionComputeResources(ModuleBuilder &module);

} // namespace detail
} // namespace loom::adg

#endif // LOOM_ADG_BUILDER_INTERNAL_H
