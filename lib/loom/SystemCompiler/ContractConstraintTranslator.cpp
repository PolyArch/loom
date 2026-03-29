#include "loom/SystemCompiler/ContractConstraintTranslator.h"

namespace loom {

//===----------------------------------------------------------------------===//
// Edge constraint translation
//===----------------------------------------------------------------------===//

std::vector<TranslatedConstraint>
translateEdgeConstraints(const TDCEdgeSpec &edgeSpec) {
  std::vector<TranslatedConstraint> constraints;
  std::string edgeLabel =
      edgeSpec.producerKernel + "->" + edgeSpec.consumerKernel;

  if (edgeSpec.ordering.has_value()) {
    TranslatedConstraint c;
    c.label = "ordering:" + std::string(orderingToString(*edgeSpec.ordering)) +
              ":" + edgeLabel;
    c.dimension = "ordering";
    c.enumValue = orderingToString(*edgeSpec.ordering);
    constraints.push_back(std::move(c));
  }

  if (edgeSpec.throughput.has_value()) {
    TranslatedConstraint c;
    c.label = "throughput:" + edgeLabel;
    c.dimension = "throughput";
    c.expression = *edgeSpec.throughput;
    constraints.push_back(std::move(c));
  }

  if (edgeSpec.placement.has_value()) {
    TranslatedConstraint c;
    c.label =
        "placement:" +
        std::string(placementToString(*edgeSpec.placement)) + ":" + edgeLabel;
    c.dimension = "placement";
    c.enumValue = placementToString(*edgeSpec.placement);
    constraints.push_back(std::move(c));
  }

  if (edgeSpec.shape.has_value()) {
    TranslatedConstraint c;
    c.label = "shape:" + edgeLabel;
    c.dimension = "shape";
    c.expression = *edgeSpec.shape;
    constraints.push_back(std::move(c));
  }

  return constraints;
}

//===----------------------------------------------------------------------===//
// Path constraint translation
//===----------------------------------------------------------------------===//

std::vector<TranslatedConstraint>
translatePathConstraints(const TDCPathSpec &pathSpec) {
  std::vector<TranslatedConstraint> constraints;

  if (!pathSpec.latency.empty()) {
    std::string pathLabel = pathSpec.startProducer + "->" +
                            pathSpec.startConsumer + "..." +
                            pathSpec.endProducer + "->" + pathSpec.endConsumer;
    TranslatedConstraint c;
    c.label = "latency:" + pathLabel;
    c.dimension = "latency";
    c.expression = pathSpec.latency;
    constraints.push_back(std::move(c));
  }

  return constraints;
}

//===----------------------------------------------------------------------===//
// Batch translation
//===----------------------------------------------------------------------===//

std::vector<TranslatedConstraint>
translateAllConstraints(const std::vector<TDCEdgeSpec> &edges,
                        const std::vector<TDCPathSpec> &paths) {
  std::vector<TranslatedConstraint> all;

  for (const auto &edge : edges) {
    auto edgeConstraints = translateEdgeConstraints(edge);
    all.insert(all.end(), std::make_move_iterator(edgeConstraints.begin()),
               std::make_move_iterator(edgeConstraints.end()));
  }

  for (const auto &path : paths) {
    auto pathConstraints = translatePathConstraints(path);
    all.insert(all.end(), std::make_move_iterator(pathConstraints.begin()),
               std::make_move_iterator(pathConstraints.end()));
  }

  return all;
}

//===----------------------------------------------------------------------===//
// Legacy conversion
//===----------------------------------------------------------------------===//

TDCEdgeSpec contractSpecToEdgeSpec(const ContractSpec &legacy) {
  TDCEdgeSpec spec;
  spec.producerKernel = legacy.producerKernel;
  spec.consumerKernel = legacy.consumerKernel;
  spec.dataTypeName = legacy.dataTypeName;

  // Map legacy ordering
  spec.ordering = legacy.ordering;

  // Map legacy visibility (Placement alias) to placement
  spec.placement = legacy.visibility;

  // No direct throughput or shape mapping from ContractSpec
  return spec;
}

} // namespace loom
