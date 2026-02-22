//===-- html_mapped_self_contained.cpp - Mapped HTML self-containment test ---===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify mapped HTML generation: output is self-contained (no CDN refs),
// contains all required viewer elements, and embeds all three DOT sources.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Viz/DOTExporter.h"

#include <string>

int main() {
  // Minimal DOT sources for all three panels.
  std::string overlay_dot = "digraph MappedOverlay { a -> b; }";
  std::string dfg_dot = "digraph MappedDFG { sw_0 -> sw_1; }";
  std::string adg_dot = "digraph MappedADG { hw_0 -> hw_1; }";
  std::string mapping_json =
      "{ \"swToHw\": {\"sw_0\":\"hw_0\"}, \"hwToSw\": {\"hw_0\":[\"sw_0\"]}, "
      "\"routes\": {} }";
  std::string metadata_json = "{}";

  // Generate mapped HTML.
  std::string html = loom::viz::generateMappedHTML(
      overlay_dot, dfg_dot, adg_dot, mapping_json, metadata_json,
      "Test Mapped View");

  // Basic HTML structure.
  TEST_CONTAINS(html, "<!DOCTYPE html>");
  TEST_CONTAINS(html, "<html>");
  TEST_CONTAINS(html, "</html>");
  TEST_CONTAINS(html, "Test Mapped View");

  // Self-containment: no external CDN references.
  TEST_NOT_CONTAINS(html, "unpkg.com");
  TEST_NOT_CONTAINS(html, "jsdelivr.net");
  TEST_NOT_CONTAINS(html, "cdn.");
  TEST_NOT_CONTAINS(html, "cdnjs.");

  // All three DOT sources embedded.
  TEST_CONTAINS(html, "MappedOverlay");
  TEST_CONTAINS(html, "MappedDFG");
  TEST_CONTAINS(html, "MappedADG");

  // Viewer elements.
  TEST_CONTAINS(html, "id=\"toolbar\"");
  TEST_CONTAINS(html, "id=\"graph-area\"");
  TEST_CONTAINS(html, "id=\"graph-left\"");
  TEST_CONTAINS(html, "id=\"graph-right\"");
  TEST_CONTAINS(html, "id=\"detail-panel\"");

  // Inline renderer present.
  TEST_CONTAINS(html, "renderDot");
  TEST_CONTAINS(html, "DotRenderer");

  // Mapping data embedded.
  TEST_CONTAINS(html, "mappingData");
  TEST_CONTAINS(html, "swToHw");
  TEST_CONTAINS(html, "hwToSw");

  // Viz level set to mapped.
  TEST_CONTAINS(html, "vizLevel");
  TEST_CONTAINS(html, "\"mapped\"");

  // Mode toggle for side-by-side view.
  TEST_CONTAINS(html, "btn-mode-toggle");

  return 0;
}
