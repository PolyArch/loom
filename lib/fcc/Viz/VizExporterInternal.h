#ifndef FCC_VIZ_VIZEXPORTER_INTERNAL_H
#define FCC_VIZ_VIZEXPORTER_INTERNAL_H

#include "fcc/Viz/VizExporter.h"

#include "fcc/Dialect/Fabric/FabricDialect.h"
#include "fcc/Dialect/Fabric/FabricOps.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fcc {
namespace viz_detail {

// ---- String escaping helpers ----

std::string jsonEsc(llvm::StringRef s);
std::string htmlEsc(llvm::StringRef s);
std::string scriptSafe(const std::string &s);
std::string printType(mlir::Type type);

// ---- DFG metadata helpers ----

std::string dfgEdgeType(mlir::Type type);
std::string dfgOperandName(mlir::Operation *op, unsigned idx);
std::string dfgResultName(mlir::Operation *op, unsigned idx);

// ---- Name/title helpers ----

std::string getADGName(mlir::ModuleOp topModule);
std::string getDFGName(mlir::ModuleOp topModule);
std::string makeVizTitle(mlir::ModuleOp adgModule, mlir::ModuleOp dfgModule,
                         bool hasMapping);
std::string loadVizSidecarJson(mlir::ModuleOp adgModule,
                                llvm::StringRef adgSourcePath);

// ---- JSON utility ----

unsigned getJsonUnsigned(const llvm::json::Object &obj, llvm::StringRef key,
                         unsigned defaultValue = 0);

// ---- Auto-layout structs ----

struct AutoLayoutComponent {
  std::string name;
  std::string kind;
  unsigned numInputs = 0;
  unsigned numOutputs = 0;
  unsigned fuCount = 0;
  double width = 160.0;
  double height = 80.0;
};

struct AutoLayoutConnection {
  std::string from;
  unsigned fromIdx = 0;
  std::string to;
  unsigned toIdx = 0;
};

struct AutoLayoutPlacement {
  double centerX = 0.0;
  double centerY = 0.0;
  bool valid = false;
};

struct AutoLayoutRoutePt {
  double x = 0.0;
  double y = 0.0;
};

struct AutoLayoutModuleBounds {
  double x = 0.0;
  double y = 0.0;
  double w = 0.0;
  double h = 0.0;
  bool valid = false;
};

// ---- DFG model structs ----

struct DFGJsonPortInfo {
  unsigned index = 0;
  std::string name;
  std::string type;
};

struct DFGJsonNodeInfo {
  int id = 0;
  std::string kind;
  std::string label;
  std::string display;
  std::string op;
  int argIndex = -1;
  int resultIndex = -1;
  std::string name;
  std::string type;
  std::vector<DFGJsonPortInfo> inputs;
  std::vector<DFGJsonPortInfo> outputs;
  double width = 0.0;
  double height = 0.0;
  double x = 0.0;
  double y = 0.0;
  bool hasLayout = false;
};

struct DFGJsonEdgeInfo {
  int id = 0;
  int from = 0;
  unsigned fromPort = 0;
  int to = 0;
  unsigned toPort = 0;
  std::string edgeType;
  std::string valueType;
  std::vector<std::pair<double, double>> points;
};

struct DFGJsonModel {
  std::string funcName;
  std::vector<DFGJsonNodeInfo> nodes;
  std::vector<DFGJsonEdgeInfo> edges;
};

// ---- ADG auto-layout functions (VizExporterADG.cpp) ----

AutoLayoutComponent parseAutoLayoutComponent(const llvm::json::Object &obj);
double computeModuleMarginForAreaRatio(double contentW, double contentH,
                                       double areaRatio);
std::array<unsigned, 4> buildAutoLayoutPortSideCounts(unsigned count,
                                                      unsigned sideCount);
std::array<unsigned, 2> buildAutoLayoutPEPortSideCounts(unsigned count);
AutoLayoutRoutePt computeAutoLayoutInputPortPos(
    const AutoLayoutComponent &comp, const AutoLayoutPlacement &placement,
    unsigned portIdx);
AutoLayoutRoutePt computeAutoLayoutOutputPortPos(
    const AutoLayoutComponent &comp, const AutoLayoutPlacement &placement,
    unsigned portIdx);
AutoLayoutModuleBounds
computeAutoLayoutModuleBounds(const std::vector<AutoLayoutComponent> &components,
                              const std::vector<AutoLayoutPlacement> &placements);
AutoLayoutRoutePt computeAutoLayoutModuleInputPortPos(
    const AutoLayoutModuleBounds &bounds, unsigned numModuleInputs,
    unsigned portIdx);
AutoLayoutRoutePt computeAutoLayoutModuleOutputPortPos(
    const AutoLayoutModuleBounds &bounds, unsigned numModuleOutputs,
    unsigned portIdx);
std::string buildAutoLayoutJson(
    const std::vector<AutoLayoutComponent> &components,
    const std::vector<AutoLayoutConnection> &connections,
    unsigned numModuleInputs, unsigned numModuleOutputs,
    const std::vector<AutoLayoutPlacement> &placements);
std::string buildNeatoLayoutJsonFromADGJson(llvm::StringRef adgJsonText);
std::string selectVizLayoutJson(mlir::ModuleOp adgModule,
                                llvm::StringRef adgSourcePath,
                                llvm::StringRef adgJson,
                                VizLayoutMode layoutMode);

// ---- ADG serialization (VizExporterADG.cpp) ----

void writeADGJson(llvm::raw_ostream &os, mlir::ModuleOp topModule,
                  mlir::MLIRContext *ctx);

// ---- DFG functions (VizExporterDFG.cpp) ----

std::pair<double, double> getDFGPortAnchor(const DFGJsonNodeInfo &node,
                                           bool isInput, unsigned portIndex);
void simplifyDFGPolyline(std::vector<std::pair<double, double>> &points);
std::vector<std::pair<double, double>>
buildFallbackDFGEdgePolyline(const DFGJsonModel &model,
                              const DFGJsonEdgeInfo &edge);
std::pair<double, double> estimateDFGNodeSize(const DFGJsonNodeInfo &node);
bool routeDFGEdgesOffline(DFGJsonModel &model);
void buildDFGJsonModel(DFGJsonModel &model, mlir::ModuleOp topModule);
bool applyDotLayoutToDFGModel(DFGJsonModel &model);
void writeDFGJson(llvm::raw_ostream &os, mlir::ModuleOp topModule);

} // namespace viz_detail
} // namespace fcc

#endif // FCC_VIZ_VIZEXPORTER_INTERNAL_H
