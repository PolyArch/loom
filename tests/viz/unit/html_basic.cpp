//===-- html_basic.cpp - HTML viewer generation test ---------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify HTML generation: output contains required elements (DOT source,
// toolbar, graph area, detail panel) per spec-viz-gui.md.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Viz/DOTExporter.h"

#include <string>

int main() {
  // Simple DOT input.
  std::string dot = "digraph G { a -> b; }";

  // Generate HTML.
  std::string html =
      loom::viz::generateHTML(dot, "Test Visualization");

  // Verify HTML structure per spec-viz-gui.md.
  TEST_CONTAINS(html, "<!DOCTYPE html>");
  TEST_CONTAINS(html, "<html>");
  TEST_CONTAINS(html, "</html>");
  TEST_CONTAINS(html, "Test Visualization");

  // Toolbar elements.
  TEST_CONTAINS(html, "id=\"toolbar\"");
  TEST_CONTAINS(html, "btn-zoom-in");
  TEST_CONTAINS(html, "btn-zoom-out");
  TEST_CONTAINS(html, "btn-fit");
  TEST_CONTAINS(html, "btn-mode-toggle");

  // Graph area.
  TEST_CONTAINS(html, "id=\"graph-area\"");
  TEST_CONTAINS(html, "id=\"graph-left\"");
  TEST_CONTAINS(html, "id=\"graph-right\"");

  // Detail panel.
  TEST_CONTAINS(html, "id=\"detail-panel\"");
  TEST_CONTAINS(html, "id=\"detail-content\"");
  TEST_CONTAINS(html, "detail-close");

  // Embedded DOT source.
  TEST_CONTAINS(html, "dotSources");
  TEST_CONTAINS(html, "digraph G");

  // viz.js reference (CDN or inlined).
  TEST_CONTAINS(html, "viz");

  // CSS styles.
  TEST_CONTAINS(html, "<style>");
  TEST_CONTAINS(html, ".highlight");

  return 0;
}
