#ifndef LOOM_LIB_FABRIC_VISUALIZATION_FABRICVISUALIZATIONINTERNAL_H
#define LOOM_LIB_FABRIC_VISUALIZATION_FABRICVISUALIZATIONINTERNAL_H

#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <string>
#include <vector>

namespace loom::fabric::visualization {

struct Point final {
  double x = 0.0;
  double y = 0.0;
};

struct Node final {
  std::string id;
  std::string label;
  std::string detail;
  std::string kind;
  double width = 176.0;
  double height = 72.0;
  double x = 0.0;
  double y = 0.0;
};

struct Edge final {
  std::size_t source = 0;
  std::size_t destination = 0;
  std::string label;
  std::string kind;
  std::vector<Point> route;
};

struct Graph final {
  std::string id;
  std::string title;
  std::string subtitle;
  std::string kind;
  std::string artifactIdentity;
  std::vector<Node> nodes;
  std::vector<Edge> edges;
  double width = 0.0;
  double height = 0.0;
};

struct Document final {
  std::string title;
  std::string rootIdentity;
  std::vector<Graph> graphs;
};

llvm::Expected<Document> buildDocument(const FinalizedFabricRoot &root,
                                       const ArtifactStore &store);

void computeLayeredLayout(Graph &graph);

llvm::Error writeHtml(const Document &document, llvm::raw_ostream &output);

} // namespace loom::fabric::visualization

#endif // LOOM_LIB_FABRIC_VISUALIZATION_FABRICVISUALIZATIONINTERNAL_H
