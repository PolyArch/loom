# Mapping Visualization Metadata

This document specifies optional visualization metadata for Loom mapping
artifacts.

Visualization metadata supports GUI rendering, human inspection, and DSE
comparison. It does not affect software semantics, hardware semantics,
mapping legality, simulator behavior, runtime behavior, RTL lowering, or
FPA estimation.

## Visualization Record Families

Visualization metadata may include:

* layout references;
* view definitions;
* overlays;
* style classes;
* labels and grouping;
* metric display hints.

Every visualization record is optional. A tool must be able to render a
mapping artifact as an arbitrary graph without visualization metadata.

## Layout Reference

A layout-reference record connects mapping visualization to optional
layout metadata declared by Fabric ADG.

Required fields:

* `record_id`.
* `layout_ref`: hardware reference to a visualization layout declared
  by Fabric ADG.
* `scope`: system, module, thread, graph, route, memory, or custom.

Optional fields:

* `projection`: `graph`, `grid1d`, `grid2d`, `grid3d`, `tree`,
  `hierarchy`, or `custom`.
* `camera`.
* `rank_order`.
* `custom_payload`.

Layouts and coordinates are metadata. Connectivity and routing still
come from Fabric links and mapping route records.

## View Definition

A view-definition record names a GUI view over the mapping artifact.

Required fields:

* `record_id`.
* `view_kind`: `system`, `thread`, `graph`, `route`, `memory`,
  `schedule`, `resource`, `diagnostic`, or `comparison`.
* `subjects`: software, hardware, or mapping references included in the
  view.

Optional fields:

* `layout`.
* `filters`.
* `sort`.
* `group_by`.
* `metric_axes`.

View definitions are hints. GUI tools may choose a different rendering
when needed.

## Overlay

An overlay record connects software objects, hardware objects, and
mapping records for display.

Required fields:

* `record_id`.
* `overlay_kind`: `placement`, `route`, `schedule`, `memory`,
  `diagnostic`, `metric`, or `comparison`.
* `subjects`: non-empty list of references.

Optional fields:

* `style_class`.
* `label`.
* `visibility`.
* `metric_ref`.
* `tooltip_fields`.

Placement overlays may draw software objects on hardware layout
positions. Route overlays may draw paths through explicit route
segments. Schedule overlays may draw resource use over time or tag
domains.

## Style Class

A style-class record names display styling without embedding GUI logic.

Required fields:

* `record_id`.
* `class_name`.

Optional fields:

* `color`.
* `line_style`.
* `fill`.
* `priority`.
* `semantic_role`.

Style classes are advisory. They must not be used by verifiers or
simulators to infer mapping semantics.

## Labels and Grouping

Labels and groups help humans inspect large mappings.

Required fields:

* `record_id`.
* `subjects`: non-empty list of references.

Optional fields:

* `label`.
* `group`.
* `collapsed_by_default`.
* `description`.

Labels may use human-readable names. Tools that need stable identity
must use references, not labels.

## Validation

The visualization verifier checks:

* every visualization subject resolves;
* every referenced layout exists;
* coordinate rank matches the referenced layout when coordinates are
  present;
* overlays reference mapping records compatible with their
  `overlay_kind`;
* visualization metadata does not introduce extra routes, placements,
  hardware resources, or software operations.

Invalid visualization metadata may be rejected without rejecting the
base mapping when the consumer does not require visualization. A GUI
consumer may reject the visualization profile while still showing a
fallback graph view of the base artifact.

## Acceptance Criteria

Visualization metadata is complete when:

* arbitrary-topology systems can be rendered using graph layout without
  optional coordinates;
* regular topologies such as mesh-like or stacked-grid systems can be
  rendered through optional layout metadata;
* route, placement, memory, schedule, diagnostic, and metric overlays
  can reference the same base mapping records;
* removing all visualization records leaves the mapping legality and
  simulator behavior unchanged.
